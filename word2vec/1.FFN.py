#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:47
# @Author  : JZJ
# @File    : 1.FFN.py
# @Software: win11 python3.7

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils import BOS_TOKEN, EOS_TOKEN, load_crawldata
from utils import load_reuters, save_pretrained, get_loader, init_weights

class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        self.data=[]
        self.bos=vocab[BOS_TOKEN]
        self.eos=vocab[EOS_TOKEN]
        for sentence in tqdm(corpus,desc='Dataset Construtions'):
            sentence=[self.bos]+sentence+[self.eos]
            if len(sentence) <context_size:
                continue
            for i in range(context_size,len(sentence)):
                context=sentence[i-context_size:i]
                target=sentence[i]
                # n元语言模型
                self.data.append((context,target))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self,examples):
        inputs=torch.tensor([ex[0] for ex in examples],dtype=torch.long)
        targets=torch.tensor([ex[1] for ex in examples],dtype=torch.long)
        return (inputs,targets)

class FeedForwardNNLM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,context_size,hidden_dim):
        super(FeedForwardNNLM, self).__init__()
        #词向量层
        self.embeddings=nn.Embedding(vocab_size,embedding_dim)
        #线性变换
        self.linear1=nn.Linear(context_size*embedding_dim,hidden_dim)

        self.linear2=nn.Linear(hidden_dim,vocab_size)
        self.activate=F.relu

    def forward(self,inputs):
        embeds=self.embeddings(inputs).view((inputs.shape[0],-1))
        hidden=self.activate(self.linear1(embeds))
        output=self.linear2(hidden)

        log_probs=F.log_softmax(output,dim=1)
        return log_probs

embedding_dim = 128
context_size = 2
hidden_dim = 64
batch_size = 1024
num_epoch = 30

corpus,vocab=load_crawldata()
datasets=NGramDataset(corpus,vocab,context_size)
data_loader=get_loader(datasets,batch_size)

model=FeedForwardNNLM(len(vocab.idx_to_token),embedding_dim,context_size,hidden_dim)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 优化器和损失函数
optimizer=optim.Adam(model.parameters(),lr=0.001)
nll_loss=nn.NLLLoss()

model.train()
total_losses=[]
print("3.训练词向量")
for epoch in range(num_epoch):
    total_loss=0
    for batch in tqdm(data_loader,desc=f"Training Epoch {epoch+1}"):
        inputs,targets=[x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs=model(inputs)
        loss=nll_loss(log_probs,targets)
        loss.backward()
        optimizer.step()
        # item得到的是数值类型
        total_loss+=loss.item()
    print(f"Loss:{total_loss:.2f}")
    total_losses.append(total_loss)

# 保存词向量（model.embeddings）
save_pretrained(vocab, model.embeddings.weight.data, "ffn_{:.4f}.vec".format(total_losses[-1]),total_losses)

