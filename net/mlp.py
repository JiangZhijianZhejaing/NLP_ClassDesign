#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 21:02
# @Author  : JZJ
# @File    : mlp.py
# @Software: win11 python3.7
import torch.nn as nn
import matplotlib.pyplot as plt # 画图库
import torch
from torch import nn, optim
from torch.nn import functional as F, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_class=0):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def forward(self, inputs, labels=None):
        embedding = self.embedding(inputs)
        # shape:batch_size,seqence_length,length
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        # log_probs = F.log_softmax(outputs, dim=1)
        loss=0
        if labels is not None:
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        dic={'loss':loss,'logits':outputs}
        return dic

    def init_weights(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.Linear):
                nn.init.normal_(ly.weight, std=0.01)

if __name__ == '__main__':
    model=MLP(782,123,123,23)
    input = torch.randint(1, 783, (5, 10))
    model(input)
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print(num_parameters)
