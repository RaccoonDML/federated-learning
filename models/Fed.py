#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_weighted(w,losses):
    num=len(w)
    losses=1/losses
    print(losses)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():#对每一层的参数
        #开始累加
        for i in range(1,num):#对每个client端
            w_avg[k] +=losses[i]*w[i][k]
        w_avg[k] = torch.div(w_avg[k], sum(losses))
    return w_avg