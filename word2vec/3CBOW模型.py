#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 16:21
# @Author  : JZJ
# @File    : 3CBOW模型.py
# @Software: win11 python3.7

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_crawldata
from utils import load_reuters, save_pretrained, get_loader, init_weights

class CbowDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            sentence = [self.bos] + sentence+ [self.eos]
            if len(sentence) < context_size * 2 + 1:
                continue
            for i in range(context_size, len(sentence) - context_size):
                # 模型输入：左右分别取context_size长度的上下文
                context = sentence[i-context_size:i] + sentence[i+1:i+context_size+1]
                # 模型输出：当前词
                target = sentence[i]
                # 注意
                self.data.append((context, target))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]
    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return (inputs, targets)
#模型
class CbowModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CbowModel, self).__init__()
        #词向量层
        self.embeddings=nn.Embedding(vocab_size,embedding_dim)
        # 输出层
        self.output=nn.Linear(embedding_dim,vocab_size,bias=False)

    def forward(self,inputs):
        # 1024,4 -> 1024,4,hidden_size
        embeds=self.embeddings(inputs)
        # 对上下层取平均->1024,hidden_size
        hidden=embeds.mean(dim=1)
        # 1024,hidden_size->1024,31081
        output=self.output(hidden)
        # 限定运算的维度
        log_probs=F.log_softmax(output,dim=-1)
        return log_probs

embedding_dim=128
context_size=2
hidden_dim=128
batch_size=1024
num_epoch=30

corpus,vocab=load_crawldata()
dataset=CbowDataset(corpus,vocab,context_size=context_size)
data_loader=get_loader(dataset,batch_size)

#定义模型加载到设备
model=CbowModel(len(vocab.idx_to_token),embedding_dim)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 优化器和损失函数
optimizer=optim.Adam(model.parameters(),lr=0.001)
nll_loss=nn.NLLLoss()
# 迭代训练
model.train()
total_losses = []
for epoch in range(num_epoch):
    total_loss=0
    for batch in tqdm(data_loader,desc=f"Training Epoch {epoch}"):
        input,targets=[x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs=model(input)
        loss=nll_loss(log_probs,targets)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Loss:{total_loss:2f}")
    total_losses.append(total_loss)
save_pretrained(vocab,model.embeddings.weight.data,"cbow_{:.4f}.vec".format(total_losses[-1]),total_losses)


# nll_loss=nn.NLLLoss()
# #构造CBOW模型，加载到device
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# model=CbowModel(len(vocab),embedding_dim)
# model.to(device)
# optimizer=optim.Adam(model.parameters(),lr=0.001)
# model.train()
# for epoch in range(num_epoch):
#     total_loss=0
#     for batch in tqdm(data_loader,desc=f'Training Epoch {epoch}'):
#         inputs,targets=[x.to(device) for x in batch]
#         optimizer.zero_grad()
#         #先softmax，然后log函数，最后nll_loss 本质上是交叉熵函数
#         log_probs=model(inputs)
#         loss=nll_loss(log_probs,targets)
#         loss.backward()
#         optimizer.step()
#         total_loss+=loss.item()
#     print(f"Loss:{total_loss:.2f}")
#
# # 保存词向量（model.embeddings）
# save_pretrained(vocab, model.embeddings.weight.input, "cbow.vec")