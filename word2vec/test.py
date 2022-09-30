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

