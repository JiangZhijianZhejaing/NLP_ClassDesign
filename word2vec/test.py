#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 9:05
# @Author  : JZJ
# @File    : test.py
# @Software: win11 python3.7
import re
strs=[]
data=[]
with open("../data/crawldata.txt", encoding='utf-8') as f:
    data=f.readlines()
    data=data[:int(len(data) * 0.1)]
for str_ in data:
    if str_=="\n": continue
    str_ = re.sub(r'\d +', '', str_)
    table=str.maketrans({'[':'','!':'','"':'','#':'','$':'','%':'','{':'','}':'','~':'',']':'','<':'','>':'','n':''})
    str_ = str_.translate(table)
    str_=str_.strip()
    str_ = [one for one in str_]
    strs.append(str_)
print()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils import BOS_TOKEN, EOS_TOKEN, load_crawldata
from utils import load_reuters, save_pretrained, get_loader, init_weights



"""
《长恨歌》——白居易
汉皇重色思倾国，御宇多年求不得。杨家有女初长成，养在深闺人未识。天生丽质难自弃，一朝选在君王侧。
回眸一笑百媚生，六宫粉黛无颜色。春寒赐浴华清池，温泉水滑洗凝脂。侍儿扶起娇无力，始是新承恩泽时。
云鬓花颜金步摇，芙蓉帐暖度春宵。春宵苦短日高起，从此君王不早朝。承欢侍宴无闲暇，春从春游夜专夜。
后宫佳丽三千人，三千宠爱在一身。金屋妆成娇侍夜，玉楼宴罢醉和春。姊妹弟兄皆列土，可怜光彩生门户。
遂令天下父母心，不重生男重生女。骊宫高处入青云，仙乐风飘处处闻。缓歌慢舞凝丝竹，尽日君王看不足。
渔阳鼙鼓动地来，惊破霓裳羽衣曲。九重城阙烟尘生，千乘万骑西南行。翠华摇摇行复止，西出都门百余里。
六军不发无奈何，宛转蛾眉马前死。花钿委地无人收，翠翘金雀玉搔头。君王掩面救不得，回看血泪相和流。
黄埃散漫风萧索，云栈萦纡登剑阁。峨嵋山下少人行，旌旗无光日色薄。蜀江水碧蜀山青，圣主朝朝暮暮情。
行宫见月伤心色，夜雨闻铃肠断声。天旋地转回龙驭，到此踌躇不能去。马嵬坡下泥土中，不见玉颜空死处。
君臣相顾尽沾衣，东望都门信马归。归来池苑皆依旧，太液芙蓉未央柳。芙蓉如面柳如眉，对此如何不泪垂。
春风桃李花开日，秋雨梧桐叶落时。西宫南内多秋草，落叶满阶红不扫。梨园弟子白发新，椒房阿监青娥老。
夕殿萤飞思悄然，孤灯挑尽未成眠。迟迟钟鼓初长夜，耿耿星河欲曙天。鸳鸯瓦冷霜华重，翡翠衾寒谁与共。
悠悠生死别经年，魂魄不曾来入梦。临邛道士鸿都客，能以精诚致魂魄。为感君王辗转思，遂教方士殷勤觅。
排空驭气奔如电，升天入地求之遍。上穷碧落下黄泉，两处茫茫皆不见。忽闻海上有仙山，山在虚无缥渺间。
楼阁玲珑五云起，其中绰约多仙子。中有一人字太真，雪肤花貌参差是。金阙西厢叩玉扃，转教小玉报双成。
闻道汉家天子使，九华帐里梦魂惊。揽衣推枕起徘徊，珠箔银屏迤逦开。云鬓半偏新睡觉，花冠不整下堂来。
风吹仙袂飘飖举，犹似霓裳羽衣舞。玉容寂寞泪阑干，梨花一枝春带雨。含情凝睇谢君王，一别音容两渺茫。
昭阳殿里恩爱绝，蓬莱宫中日月长。回头下望人寰处，不见长安见尘雾。惟将旧物表深情，钿合金钗寄将去。
钗留一股合一扇，钗擘黄金合分钿。但教心似金钿坚，天上人间会相见。临别殷勤重寄词，词中有誓两心知。
七月七日长生殿，夜半无人私语时。在天愿作比翼鸟，在地愿为连理枝。天长地久有时尽，此恨绵绵无绝期。
"""
#
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPool1D, GlobalMaxPool1D, Dense
# from warnings import filterwarnings
# filterwarnings('ignore')  # 不打印警告
#
# """配置"""
# window = 32  # 滑窗大小
# filters = 25  # 卷积录波器数量
# kernel_size = 5  # 卷积核大小
# times = 20  # 训练总次数
# batch_size = 2
# epochs = 100
# __doc__="《长恨歌》——白居易\n汉皇重色思倾国，御宇多年求不得。杨家有女初长成，养在深闺人未识。天生丽质难自弃，一朝选在君王侧。\n回眸一笑百媚生，六宫粉黛无颜色。春寒赐浴华清池，温泉水滑洗凝脂。侍儿扶起娇无力，始是新承恩泽时。"
#
# """语料加载"""
# seq_chr = __doc__.replace('\n', '').replace('《长恨歌》——白居易', '')
#
# """数据预处理"""
# len_seq = len(seq_chr)
# chr_set = set(seq_chr)  # 字库
# len_chr = len(chr_set)
# print('语料长度', len_seq, '字汇量', len_chr)
# chr2id = {c: i for i, c in enumerate(chr_set)}
# id2chr = {i: c for c, i in chr2id.items()}
# seq_id = [chr2id[c] for c in seq_chr]  # 文字序列 --> 索引序列
#
# """输入层和标签"""
# reshape = lambda x: np.reshape(x, (-1, window, 1)) / len_chr
# x = np.array([seq_id[i: i + window] for i in range(len_seq - window)])
# x = reshape(x)
# y =  np.array([seq_id[i + window] for i in range(len_seq - window)],dtype=np.uint8)
#
#
# """建模"""
# model = Sequential()
# model.add(Conv1D(filters, kernel_size * 3, activation='relu'))
# model.add(MaxPool1D())
# model.add(Conv1D(filters * 2, kernel_size, activation='relu'))
# model.add(GlobalMaxPool1D())
# model.add(Dense(len_chr, activation='softmax'))
# model.compile('adam', 'categorical_crossentropy')
#
# """随机采样"""
# def draw_sample(predictions, temperature):
#     pred = predictions.astype('float64')  # 提高精度防报错
#     pred = np.log(pred) / temperature
#     pred = np.exp(pred)
#     pred = pred / np.sum(pred)
#     pred = np.random.multinomial(1, pred, 1)
#     return np.argmax(pred)
#
# """预测"""
# def predict(t, pred=None):
#     if pred is None:
#         randint = np.random.randint(len_seq - window)
#         pred = seq_id[randint: randint + window]
#     if t:
#         print('随机采样，温度：%.1f' % t)
#         sample = draw_sample
#     else:
#         print('贪婪采样')
#         sample = np.argmax
#     for _ in range(window):
#         x_pred = reshape(pred[-window:])
#         y_pred = model.predict(x_pred)[0]
#         i = sample(y_pred, t)
#         pred.append(i)
#     text = ''.join([id2chr[i] for i in pred[-window:]])
#     print('\033[033m%s\033[0m' % text)
#
# """训练"""
# for e in range(times):
#     model.fit(x, y, batch_size, epochs, verbose=0)
#     print(str(e + 1).center(window * 2, '-'))
#     # 训练效果展示
#     for t in (None, 1, 1.5, 2):
#         predict(t)
# print('complete'.center(window * 2, '-'))
# for t in (None, 1, 1.5, 2):
#     predict(t, seq_id[-window:])



