#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:47
# @Author  : JZJ
# @File    : 2.RNN.py
# @Software: win11 python3.7
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_crawldata
from utils import load_reuters, save_pretrained, get_loader, init_weights


class RnnlmDataset(Dataset):
    def __init__(self,corpus, vocab):
        self.data=[]
        self.bos=vocab[BOS_TOKEN]
        self.eos=vocab[EOS_TOKEN]
        self.pad=vocab[PAD_TOKEN]

        for sentence in tqdm(corpus,desc="Dataset Construction"):
            input=[self.bos]+sentence
            targets=sentence+[self.eos]
            self.data.append((input,targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self,examples):
        inputs=[torch.tensor(ex[0][:100])  if len(ex[0])>100 else torch.tensor(ex[0])for ex in examples]
        targets=[torch.tensor(ex[1][:100])  if len(ex[1])>100 else torch.tensor(ex[1]) for ex in examples]
        inputs=pad_sequence(inputs,batch_first=True,padding_value=self.pad)
        targets=pad_sequence(targets,batch_first=True,padding_value=self.pad)
        return (inputs,targets)

class RNNLM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(RNNLM, self).__init__()
        #词向量层
        self.embeddings=nn.Embedding(vocab_size,embedding_dim)
        # 循环神经网络
        self.rnn=nn.LSTM(embedding_dim,hidden_dim,batch_first=True)
        # 输出层
        self.output=nn.Linear(hidden_dim,vocab_size)

    def forward(self,inputs):
        embeds=self.embeddings(inputs)
        hidden,_=self.rnn(embeds)
        output=self.output(hidden)
        log_probs=F.log_softmax(output,dim=2)
        return log_probs

embedding_dim = 128
context_size = 2
hidden_dim = 64
batch_size = 16
num_epoch = 30

corpus,vocab=load_crawldata()
datasets = RnnlmDataset(corpus, vocab)
data_loader = get_loader(datasets, batch_size)

model = RNNLM(len(vocab.idx_to_token),embedding_dim,  hidden_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
nll_loss = nn.NLLLoss()

model.train()
total_losses = []
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs.view(-1,log_probs.shape[-1]), targets.view(-1))
        loss.backward()
        optimizer.step()
        # item得到的是数值类型
        total_loss += loss.item()
    print(f"Loss:{total_loss:.2f}")
    total_losses.append(total_loss)

# 保存词向量（model.embeddings）
save_pretrained(vocab, model.embeddings.weight.data, "rnnlm_{:.4f}.vec".format(total_losses[-1]),total_losses)


