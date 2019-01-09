#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:12:14 2018

@author: xuweijia
"""
import os
import pickle
import json
import copy
import time
import numpy as np
from expand_words import make_dict,stopwords

father_file='20w'
train_data_file='train_20w_list.json'
dev_data_file='dev_20w_list.json'
test_data_file='test_20w_list.json'
data_file=[train_data_file,dev_data_file,test_data_file]

dict_file='word_dict.pkl'
r_dict_file='r_dict.pkl'

choose_top_K=15000 # not use temporarily
most_e_file='most_e_{}.pkl'.format(choose_top_K)


# vocab drop stopword,@entity,  leave only alpha
# MOST COMMON 9%-14%
word_dict,r_dict,most_common_e=make_dict(father_file,data_file,dict_file,r_dict_file,most_e_file,choose_top_K,top_K=None) # 107835
index_2_word=dict(zip(word_dict.values(),word_dict.keys()))
index_2_r=dict(zip(r_dict.values(),r_dict.keys()))

# 246423 samples

with open(os.path.join(father_file,train_data_file),'r') as f:
    samples=json.load(f)

with open(os.path.join(father_file,dev_data_file),'r') as f:
    samples+=json.load(f)
    
with open(os.path.join(father_file,test_data_file),'r') as f:
    samples+=json.load(f)

#

C=len(r_dict)
V=len(word_dict)

table_N11=np.zeros((C,V))
table_N10=np.zeros((C,V))
table_N01=np.zeros((C,V))
table_N00=np.zeros((C,V))

index_term=list(range(V))
index_c=list(range(C))

start=time.time()
for i,sample in enumerate(samples):
    print('n:{}'.format(i))
    c=sample['query']
    c_id=r_dict[c]

    doc=sample['doc'].strip().split()
    doc= [w.lower() for w in doc if not w.startswith('@entity')]
    doc=list(filter(lambda x:x.isalpha(),doc))
    doc= [w for w in doc if w not in stopwords]
    terms_id=list(map(lambda w:word_dict.get(w,0),doc))
    
#    if len(terms_id==0):
#        continue
    
    # for this class
    r_terms_id=list(set(index_term).difference(set(terms_id)))
    r_c_id=copy.deepcopy(index_c)
    r_c_id.remove(c_id)
    # N11
    table_N11[c_id,terms_id]+=1        # every class cooccur times with each term
    # N01
    table_N01[c_id,r_terms_id]+=1      # is c ,not contain term
    
    # N10  not c ,contain terms
    for index in r_c_id:
        table_N10[index,terms_id]+=1      #  contain e and are not c
    
    #N00
    for index in r_c_id:
        table_N00[index,r_terms_id]+=1    #  not contain e and are not c

end=time.time()
print('t={}h'.format((end-start)/3600))
table_all=(table_N11+table_N10+table_N01+table_N00)*((table_N11*table_N00-table_N01*table_N10))**2/((table_N11+table_N01)*(table_N11+table_N10)*(table_N00+table_N10)*(table_N00+table_N01)+0.001)

top_K=20
r_words={}
for c in range(C):
    r=index_2_r[c]
#    index=np.argsort(-table_all[c,:])[:top_K] # bigger. more likely
#    words= list(map(lambda i:index_2_word[i],index))
#    
    index=np.argsort(-table_all[c,:])
    words= list(map(lambda i:index_2_word[i],index))
    words=[w for w in words if w in most_common_e]
    words=words[:top_K]
    r_words[r]=words


with open('f_select_most_{}'.format(choose_top_K),'w') as f:
    json.dump(r_words,f)