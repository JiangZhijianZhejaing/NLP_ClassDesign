#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 17:54
# @Author  : JZJ
# @File    : transformer.py
# @Software: win11 python3.7
import torch
from nltk.corpus import sentence_polarity
from torch.utils.data import DataLoader,Dataset
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
from vocab import Vocab
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
import math
device='cpu'

def length_to_mask(lengths):
    '''
    :param lengths: 将序列长度转化为Mask矩阵
    :return:
    '''
    # max_len=torch.max(lengths)
    # max_len.to(device)
    # lengths.to(device)
    # mask=torch.arange(max_len).expand(lengths.shape[0],max_len)<lengths.unsqueeze(1)
    max_len = torch.max(lengths)
    #注意需要将模型转换到GPU
    mask = (torch.arange(max_len).expand(lengths.shape[0], max_len)).to(device) < (lengths.unsqueeze(1)).to(device)
    return mask


class PositionalEncoding(nn.Module):
    #提前判断句子的长度信息
    def __init__(self,d_model,dropout=0.1,max_len=512):
        super(PositionalEncoding, self).__init__()
        pe=torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #位置编码公式
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)  #奇数位置进行编码
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+self.pe[:x.size(0),:] #输入的词向量与位置编码相加
        return x
class Transformer(nn.Module):
    def __init__(self,vocab_size,embedding_dim,heads=4,N=4,dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 词向量
        # 位置编码
        self.position_embedding = PositionalEncoding(embedding_dim, dropout)
        # #初始化
        # nn.initializer.set_global_initializer(nn.initializer.XavierNormal(),nn.initializer.Constant(0.))
        #编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads)
        self.encoder=nn.TransformerEncoder(encoder_layer,num_layers=N)
        #解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=heads)
        self.decoder=nn.TransformerDecoder(decoder_layer, num_layers=N)
        #全连接
        self.out=nn.Linear(embedding_dim,vocab_size)
        self.init_weights()

    def forward(self,input,labels=None):
        x = self.embedding(input)
        encoder_output=self.encoder(x)
        memory=self.encoder(x)
        decoder_output=self.decoder(encoder_output,memory)
        outputs=self.out(decoder_output)
        loss=0
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
    model=Transformer(782,128,4,4)
    input = torch.randint(1,783,(5, 10 ))
    model(input)
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print(num_parameters)