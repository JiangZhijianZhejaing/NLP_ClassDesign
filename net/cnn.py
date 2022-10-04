#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 17:54
# @Author  : JZJ
# @File    : cnn.py
# @Software: win11 python3.7

import torch
from torch import nn, optim
from torch.nn import functional as F, CrossEntropyLoss

class CNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim,filter_size=5,num_filter=160):
        super(CNN, self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.conv1d=nn.Conv1d(embedding_dim,num_filter,filter_size,padding=int(filter_size/2))
        self.activate=F.relu
        self.conv1d_2 = nn.Conv1d(int(num_filter/2), num_filter*2, filter_size, padding=int(filter_size/2))
        self.activate = F.relu
        self.linear=nn.Linear(num_filter,vocab_size)
        self.init_weights()


    def forward(self,inputs,labels=None):
        embedding=self.embedding(inputs)
        '''
            输出batch_size,filter_size,length
        '''
        convolution=self.activate(self.conv1d(embedding.permute(0,2,1)))
        pooling=F.max_pool1d(convolution.permute(0,2,1),kernel_size=2)
        convolution=self.activate(self.conv1d_2(pooling.permute(0,2,1)))
        pooling=F.max_pool1d(convolution.permute(0,2,1),kernel_size=2)
        outputs=self.linear(pooling)
        loss = 0
        if labels is not None:
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        dic = {'loss': loss, 'logits': outputs}
        return dic

    def init_weights(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.Linear):
                nn.init.normal_(ly.weight, std=0.01)

if __name__ == '__main__':
    model=CNN(782,123,5,120)
    input = torch.randint(1,783,(5,10))
    model(input)
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print(num_parameters)
