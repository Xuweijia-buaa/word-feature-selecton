#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 21:45:17 2018

@author: xuweijia
"""
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# list(model.parameters())[0].cpu().data.numpy()
# np.argpartition

# self.Linear.weight.data.numpy()
class Logisit_model(nn.Module):
    def __init__(self,vocab_size):
        super(Logisit_model, self).__init__()
        self.Linear=nn.Linear(vocab_size,1)
        self.sigmoid = nn.Sigmoid()
        self.Linear.weight.requires_grad=True
        bias=np.zeros(1)
        self.Linear.bias.data.copy_(torch.from_numpy(bias))
        self.Linear.bias.requires_grad=False
        self.loss=nn.BCELoss()
        # bian
    def forward(self,x,y):
        s=self.Linear(x)           # B,V  V,1  
        predict=self.sigmoid(s).view(-1)    # B,1
        #loss0=-torch.mean(torch.mul(y,torch.log(self.sigmoid(s))) + torch.mul(1-y,torch.log(1-self.sigmoid(s))))          # assert loss!=loss0
        loss=self.loss(predict,y)
        
        # in cpu
        predict=(predict>0.5).view(-1).data              # B (not Variable)
        y=(y==1).data                                    # B (not Variable)
        
        # int
        acc=torch.sum(predict==y)            # acc/batch_size
        
        TP=torch.sum(predict & y)            # true positive  ( & only work for ByteTensors,no Variable)
        n_true=torch.sum(y)
        n_predict_true=torch.sum(predict)
        #precision = 1.0 * TP / n_predict_true          # all predict
        #recall = 1.0 * TP / n_true                     # all truth, have how many
        # f1 = (2 * precision * recall) / (precision + recall)
        return loss,acc,TP,n_true,n_predict_true
        