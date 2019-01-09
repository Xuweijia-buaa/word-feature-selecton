#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:10:23 2018

@author: xuweijia
"""
import torch
from torch.utils.data import Dataset
from expand_words import vectorize
class Data(Dataset):
    def __init__(self, samples,model_id,word_dict,r_dict):
        self.samples = samples
        self.model_id=model_id
        self.word_dict=word_dict
        self.r_dict=r_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return vectorize(self.samples[index],self.word_dict,self.r_dict,self.model_id)
    
#dataset=Data(samples,model_id,word_dict,r_dict)
#sampler = torch.utils.data.sampler.RandomSampler(dataset) # no repeat
#sampler =torch.utils.data.sampler.SequentialSampler
#data_loader = torch.utils.data.DataLoader(dataset,batch_size,sampler=sampler)
#batch_size=5
#data_loader = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

#for ex in enumerate(data_loader):
#    batch_size = ex[0].size(0)
#    x,y=ex  # x: torch.DoubleTensor of size BxV   y:torch.LongTensor of size B
#    loss_, acc_ = model(x,y) # tensor.float size 1



