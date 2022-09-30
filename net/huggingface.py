#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 18:21
# @Author  : JZJ
# @File    : huggingface.py
# @Software: win11 python3.7
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")