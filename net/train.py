#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 19:47
# @Author  : JZJ
# @File    : train.py
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--cuda', default='True',action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--vocab_path', default='../data/vocab.txt', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--model_config', default='../config/config.json', type=str, required=False,
                        help='设置模型参数')
    parser.add_argument('--train_path', default='../data/train.pkl', type=str, required=False, help='训练集路径')
    parser.add_argument('--max_len', default=150, type=int, required=False, help='训练时，输入数据的最大长度')

    parser.add_argument('--log_path', default='../log/', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--log', default=True, help="是否记录日志")
    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    # parser.add_argument('--input_len', default=200, type=int, required=False, help='输入的长度')
    parser.add_argument('--epochs', default=20, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='训练的batch size')
    parser.add_argument('--lr', default=2e-4, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument('--log_step', default=1000, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--model_type', default='transformer', type=str, required=False,
                        help='模型类型')
    parser.add_argument('--vocab_size', default=8318, type=int, required=False,
                        help='词典大小')
    parser.add_argument('--hidden_dim', default=128, type=int, required=False,
                        help='隐层大小')
    parser.add_argument('--embedding_dim', default=128 , type=int, required=False,
                        help='词嵌入大小')
    parser.add_argument('--save_model_path', default='../model', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--vec_path', default='../word2vec/', type=str, required=False,
                        help='模型输出路径')

    parser.add_argument('--pretrained_model', default='', type=str, required=False,
                        help='预训练的模型的路径')
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')
    # parser.add_argument('--label_smoothing', default=True, action='store_true', help='是否进行标签平滑')
    parser.add_argument('--val_num', type=int, default=8000, help='验证集大小')
    args = parser.parse_args()
    return args


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path+args.model_type+"_"+args.vec_type+".log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
#读入词向量
def load_pretrained(load_path):
    with open(load_path,"r",encoding='utf-8') as fin:
        #第一行为词向量的大小
        n,d=map(int,fin.readline().split())
        tokens=[]
        embeds=[]
        for line in fin:
            line=line.rstrip().split(' ')
            token,embed=line[0],list(map(float,line[1:]))
            tokens.append(token)
            embeds.append(embed)
        embeds=torch.tensor(embeds,dtype=torch.float)
    return embeds

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


def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args):
    model.train()
    device = args.device
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(tqdm(train_dataloader)):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            if args.model_type in ['mlp','cnn','lstm','transformer']:
                logits = outputs['logits']
                loss = outputs['loss']
            else:
                logits = outputs.logits
                loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            # if (batch_idx + 1) % args.log_step == 0:
            #     logger.info(
            #         "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
            #             batch_idx + 1, epoch + 1, loss.item() * args.gradient_accumulation_steps, batch_acc, scheduler.get_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    logger.info('saving model for epoch {}'.format(epoch + 1))
    if args.model_type in ['mlp', 'cnn', 'lstm', 'transformer']:
        dirs=args.save_model_path + "/" + args.model_type + 'epoch{}/'.format(epoch + 1)
        if not os.path.exists(dirs):
            os.mkdir(dirs)
        model_path = join(args.save_model_path + "/" + args.model_type + 'epoch{}/'.format(epoch + 1),
                          args.model_type + "_" + args.vec_type + '_epoch{}.pth'.format(epoch + 1))
        torch.save(model.state_dict(), model_path)
    else:
        model_path = join(args.save_model_path, args.model_type+"_"+args.vec_type+'epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)

    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, logger, epoch, args):
    logger.info("start validating")
    model.eval()
    device = args.device
    # pad_id = args.pad_id
    # sep_id = args.sep_id
    ignore_index = args.ignore_index
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                if args.model_type in ['mlp', 'cnn', 'lstm', 'transformer']:
                    loss = outputs['loss']
                else:
                    loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception


def train(model, logger, train_dataset, validate_dataset, args):
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    early_stopping = EarlyStopping(args.patience, verbose=True, save_path=args.save_model_path)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    # scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info('starting training')
    writer = SummaryWriter(log_dir=args.log_path, filename_suffix=args.model_type+"_"+args.vec_type)
    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            logger=logger, epoch=epoch, args=args)
        validate_losses.append(validate_loss)
        writer.add_scalar("train_loss_"+args.model_type+"_"+args.vec_type, train_loss, (epoch + 1))
        writer.add_scalar("validate_loss_"+args.model_type+"_"+args.vec_type, validate_loss, (epoch + 1))

        # # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        # if validate_loss < best_val_loss:
        #     best_val_loss = validate_loss
        #     logger.info('saving current best model for epoch {}'.format(epoch + 1))
        #     model_path = join(args.save_model_path+"/"+args.model_type+"_"+args.vec_type, '_min_ppl_model'.format(epoch + 1))
        #     if not os.path.exists(model_path):
        #         os.mkdir(model_path)
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     model_to_save.save_pretrained(model_path)

        #  如果patience=0,则不进行early stopping
        if args.patience == 0:
            continue
        early_stopping(validate_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))


def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        # loss = F.cross_entropy(predict_logit, target, ignore_index=pad_idx)
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


def main():
    # 初始化参数
    args = set_args()
    # 设置使用哪些显卡进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.batch_size < 2048 and args.warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')




    # 1.初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id
    # 2.加载数据集
    train_dataset, validate_dataset = load_dataset(args)
    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    for type in ['random','ffn','rnnlm','cbow','skipgram','glove']:
        args.vec_type=type

        # 创建日志对象
        logger = create_logger(args)
        # 当用户使用GPU,并且GPU可用时
        device = 'cuda:0' if args.cuda else 'cpu'
        args.device = device
        logger.info('using device:{}'.format(device))
        # 记录参数设置
        logger.info("args:{}".format(args))
        pt_embeds =None

        # 3.创建模型
        if args.model_type == 'mlp':
            model = MLP(args.vocab_size, args.embedding_dim, args.hidden_dim)
        elif args.model_type == 'cnn':
            model = CNN(args.vocab_size, args.embedding_dim)
        elif args.model_type == 'lstm':
            model = LSTM(args.vocab_size, args.embedding_dim, args.hidden_dim)
        elif args.model_type == 'transformer':
            model = Transformer(args.vocab_size, args.embedding_dim)
        else:
            # 使用GPT模型
            model_config = GPT2Config.from_json_file(args.model_config)
            model = GPT2LMHeadModel(config=model_config)
        if not type=='random' and args.model_type in ['mlp','cnn','lstm','transformer']:
            model.embedding.data = load_pretrained(args.vec_path+type+".vec")
        elif not type=='random' and args.model_type=='gpt2':
            model.transformer.wte.data= load_pretrained(args.vec_path+type+".vec")

        model = model.to(device)

        # 4.计算模型参数数量,并且训练模型
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        logger.info("Train:{}-{}".format(args.model_type, type))
        logger.info('number of model parameters: {}'.format(num_parameters))

        train(model, logger, train_dataset, validate_dataset, args)


if __name__ == '__main__':
    main()