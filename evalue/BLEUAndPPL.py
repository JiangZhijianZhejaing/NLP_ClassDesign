#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 14:47
# @Author  : JZJ
# @File    : BLEUAndPPL.py
# @Software: win11 python3.7

import argparse
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import transformers
import pickle
import sys


from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from dataset import DialogueDataset
from torch.utils.tensorboard import SummaryWriter
from cnn import CNN
from lstm import LSTM
from mlp import MLP
from transformer import Transformer

def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    # parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                     help='模型参数')
    parser.add_argument('--log_path', default='../data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='../data/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='../model/', type=str, required=False, help='对话模型路径')
    parser.add_argument('--model_type', default='mlp', type=str, required=False,
                        help='模型类型')
    parser.add_argument('--model_epoch', default='epoch20', type=str, required=False,
                        help='模型批次')
    parser.add_argument('--word2vec', default='glove', type=str, required=False,
                        help='词向量模型类型')
    parser.add_argument('--train_path', default='../data/train.pkl', type=str, required=False, help='训练集路径')
    parser.add_argument('--save_samples_path', default="result/", type=str, required=False, help="保存模型评估结果")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--vocab_size', default=8318, type=int, required=False,
                        help='词典大小')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='训练的batch size')
    parser.add_argument('--hidden_dim', default=128, type=int, required=False,
                        help='隐层大小')
    parser.add_argument('--embedding_dim', default=128 , type=int, required=False,
                        help='词嵌入大小')
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--val_num', type=int, default=8000, help='验证集大小')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    return parser.parse_args()
class BLEU():
    def __init__(self, n_gram=1):
        super().__init__()
        self.n_gram = n_gram

    def evaluate(self, candidates, references):
        ''' 计算BLEU值
        @param candidates [[str]]: 机器翻译的句子
        @param references [[str]]: 参考的句子
        @param bleu: BLEU值
        '''

        BP = 1
        bleu = np.zeros(len(candidates))
        for k, candidate in enumerate(candidates):
            candidate=list(candidate)
            r, c = 0, 0
            count = np.zeros(self.n_gram)
            count_clip = np.zeros(self.n_gram)
            count_index = np.zeros(self.n_gram)
            p = np.zeros(self.n_gram)
            for j, candidate_sent in enumerate(candidate):
                candidate_sent=list(candidate_sent)
                # 对每个句子遍历
                for i in range(self.n_gram):
                    count_, n_grams = self.extractNgram(candidate_sent, i + 1)
                    count[i] += count_
                    reference_sents = []
                    reference_sents = [reference[j] for reference in references]
                    count_clip_, count_index_ = self.countClip(reference_sents, i + 1, n_grams)
                    count_clip[i] += count_clip_
                    c += len(candidate_sent)
                    r += len(reference_sents[count_index_])
                p = count_clip / count
            rc = r / c
            if rc >= 1:
                BP = np.exp(1 - rc)
            else:
                rc = 1
            p[p == 0] = 1e-100
            p = np.log(p)
            bleu[k] = BP * np.exp(np.average(p))
        return bleu

    def extractNgram(self, candidate, n):
        ''' 抽取出n-gram
        @param candidate: [str]: 机器翻译的句子
        @param n int: n-garm值
        @return count int: n-garm个数
        @return n_grams set(): n-grams
        '''
        count = 0
        n_grams = set()
        if (len(candidate) - n + 1 > 0):
            count += len(candidate) - n + 1
        for i in range(len(candidate) - n + 1):
            n_gram = ' '.join(str(candidate[i:i + n]))
            n_grams.add(n_gram)
        return (count, n_grams)

    def countClip(self, references, n, n_gram):
        ''' 计数references中最多有多少n_grams
        @param references [[str]]: 参考译文
        @param n int: n-gram的值s
        @param n_gram set(): n-grams

        @return:
        @count: 出现的次数
        @index: 最多出现次数的句子所在文本的编号
        '''
        max_count = 0
        index = 0
        for j, reference in enumerate(references):
            reference=list(reference)
            count = 0
            for i in range(len(reference) - n + 1):
                if (' '.join(str(reference[i:i + n])) in n_gram):
                    count += 1
            if max_count < count:
                max_count = count
                index = j
        return (max_count, index)
def collate_fn(batch):
    # 对每个batch进行padding
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

def load_dataset(args):
    """
    加载训练集和验证集
    """
    train_path = args.train_path

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)

    # 划分训练集与验证集
    val_num = args.val_num
    input_list_train = input_list[val_num:]
    input_list_val = input_list[:val_num]
    # test
    # input_list_train = input_list_train[:24]
    # input_list_val = input_list_val[:24]

    train_dataset = DialogueDataset(input_list_train, args.max_len)
    val_dataset =DialogueDataset(input_list_val, args.max_len)

    return train_dataset, val_dataset

def calPPL(logits,labels):

    softmax=torch.nn.Softmax(dim=2)
    prob=softmax(logits)
    sum=0
    for i in range(labels.size()[0]):
        x=1
        for j in range(labels.size()[1]):
            x+=prob[i,j,labels[i][j]]
        x=np.power(x,-1/labels.size()[0])
        sum+=x
    return sum


def validate(args):
    _, validate_dataset = load_dataset(args)
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    bleu_ = BLEU(4)
    device = 'cpu'
    model_path = args.model_path + args.model_type + args.model_epoch + "/" + args.model_type + "_" + args.word2vec + "_" + args.model_epoch + ".pth"
    if args.model_type == 'mlp':
        model = MLP(args.vocab_size, args.embedding_dim, args.hidden_dim)
        model.load_state_dict(torch.load(model_path), strict=False)
    elif args.model_type == 'cnn':
        model = CNN(args.vocab_size, args.embedding_dim)
        model.load_state_dict(torch.load(model_path), strict=False)
    elif args.model_type == 'lstm':
        model = LSTM(args.vocab_size, args.embedding_dim, args.hidden_dim)
        model.load_state_dict(torch.load(model_path), strict=False)
    elif args.model_type == 'transformer':
        model = Transformer(args.vocab_size, args.embedding_dim)
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    device = args.device
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            total_bleu=0
            total_ppl=0
            i=0
            j=0
            for batch_idx, (input_ids, labels) in enumerate(tqdm(validate_dataloader,desc="验证集测试BLEU值:")):
                i += 0.1
                j+=args.batch_size
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                if args.model_type in ['mlp', 'cnn', 'lstm', 'transformer']:
                    logits = outputs['logits']
                else:
                    logits = outputs.logits
                total_ppl+=calPPL(logits,labels)
                x=np.expand_dims(list(np.argmax(np.array(logits[..., :-1, :]),axis=2)),axis=1)
                total_bleu+=bleu_.evaluate(x, list(np.expand_dims(np.array(labels[..., 1:]),axis=1)))


            return (total_bleu/i).mean(),total_ppl/j
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: ran out of memory\n")




def main():
    args = set_args()
    # 计算模型BLEU值大小
    x,y=validate(args)
    print(x,y)

if __name__ == '__main__':
    main()


