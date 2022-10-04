#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 13:54
# @Author  : JZJ
# @File    : vocab.py
# @Software: win11 python3.7
# 建立字典映射
import collections
from collections import defaultdict

# class Vocab:
#     def __init__(self,tokens=None):
#         self.idx_to_token=list()
#         self.token_to_idx=dict()
#
#         if tokens is not None:
#             if "<unk>" not in tokens:
#                 tokens=tokens+["<unk>"]
#             for token in tokens:
#                 self.idx_to_token.append(token)
#                 self.token_to_idx[token]=len(self.idx_to_token)-1
#             #建立隐射
#             self.unk=self.token_to_idx['<unk>']
#
#     @classmethod
#     def build(cls,text,min_freq=1,reserved_tokens=None):
#         token_freqs=defaultdict(int)
#         for sentence in text:
#             for token in sentence:
#                 token_freqs[token]+=1
#         uniq_tokens=["<unk>"]+(reserved_tokens if reserved_tokens else [])
#         uniq_tokens+=[token for token,freq in token_freqs.items() if freq>=min_freq and token!="<unk>"]
#         # cls可以在静态方法中使用，并通过cls()方法来实例化一个对象。
#         return cls(uniq_tokens)
#
#     def __len__(self):
#         '''
#         :return: 返回表大小
#         '''
#         return len(self.idx_to_token)
#
#     def __getitem__(self, token):
#         '''
#         :param token:
#         :return: 返回token索引,不存在则返回unknow的索引
#         '''
#         return self.token_to_idx.get(token,self.unk)
#
#     def convert_tokens_to_ids(self,tokens):
#         '''
#         :param tokens:
#         :return: 查找一系列语句返回索引值
#         '''
#         return [self.token_to_idx[token] for token in tokens]
#
#     def convert_ids_to_tokens(self,indices):
#         '''
#         :param tokens:
#         :return: 通过索引返回举止
#         '''
#         return [self.idx_to_token[index] for index in indices]


# 构建词表
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Loads a vocabulary file into a dictionary."""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        self.idx_to_token=list()
        self.token_to_idx=dict()

        # vocab = defaultdict()
        index = 0
        with open("../data/vocab.txt", "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                # vocab[token] = index
                tokens.append(token)
                index += 1

        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    @classmethod
    # def build(cls,text,min_freq=1,reserved_tokens=None):
    #     token_freqs=defaultdict(int)
    #     for sentence in text:
    #         for token in sentence:
    #             token_freqs[token]+=1
    #     uniq_tokens=["<unk>"]+(reserved_tokens if reserved_tokens else [])
    #     uniq_tokens+=[token for token,freq in token_freqs.items() if freq>=min_freq and token!="<unk>"]
    #     # cls可以在静态方法中使用，并通过cls()方法来实例化一个对象。
    #     return cls(uniq_tokens)
    # return cls(vocab)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # def to_tokens(self, indices):
    #     if not isinstance(indices, (list, tuple)):
    #         return self.idx_to_token[indices]
    #     return [self.idx_to_token[index] for index in indices]

    def convert_tokens_to_ids(self,tokens):
        '''
        :param tokens:
        :return: 查找一系列语句返回索引值
        '''
        return [self.token_to_idx[token] if token in self.idx_to_token else 5 for token in tokens]

    def convert_ids_to_tokens(self,indices):
        '''
        :param tokens:
        :return: 通过索引返回举止
        '''
        return [self.idx_to_token[index] if index<=8317  else '<UNK>' for index in indices ]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

#测试embedding
from torch import  nn
import torch
'''
    词表大小为8，Embedding向量维度为3
'''
# embedding=nn.Embedding(8,3)
# input=torch.tensor([[0,1,2,1],[4,6,6,7]])
# output=embedding(input)
# print(output.shape)