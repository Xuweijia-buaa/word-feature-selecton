#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:12:30 2018

@author: xuweijia
"""
import json
import nltk
import string
import os
import pickle
from collections import Counter
#father_file='10w_plain'
#top_K=200000
#train_data_file='train_10w_list.json'
#dev_data_file='dev_10w_list.json'
#test_data_file='test_10w_list.json'
#data_file=[train_data_file,dev_data_file,test_data_file]
#dict_file='word_dict.pkl'
#r_dict_file='r_dict.pkl'
english_punctuations = [',', '.', ':',"'", ';', '?', '(', ')', '...','[', ']', '&', '!', '*', '@', '#', '$', '%',"''",'``',"'s","-","--",'–','"','""','")','"-','"-','"")','"",','"".','"";','""),','"").']
letter = []
for word in string.ascii_lowercase:
    letter.append(word)
stopwords=[]
stopwords.extend(english_punctuations)
stopwords.extend(letter)
stopwords.extend(nltk.corpus.stopwords.words('english'))
kp_list=['what','which','who','whom','when','where','why','won','by','on','in','under']

for w in kp_list:
    stopwords.remove(w)

def make_dict(father_file,data_file,dict_file,r_dict_file,most_e_file,choose_top_K,top_K=None):
    w_dict_path=os.path.join(father_file,dict_file)
    r_dict_path=os.path.join(father_file,r_dict_file)
    most_e_path=os.path.join(father_file,most_e_file)
    # exist
    if os.path.exists(w_dict_path) and os.path.exists(r_dict_path) and os.path.exists(most_e_path):
        # word_dict = pickle.load(open(file, "rb"))
        print('good,dict found!!!')
        return pickle.load(open(w_dict_path, "rb")),pickle.load(open(r_dict_path, "rb")),pickle.load(open(most_e_path, "rb"))
    
    samples=[]
    with open(os.path.join(father_file,data_file[0]),'r') as f:
        samples+=json.load(f)
    with open(os.path.join(father_file,data_file[1]),'r') as f:
        samples+=json.load(f)
    with open(os.path.join(father_file,data_file[2]),'r') as f:
        samples+=json.load(f)

    r_set=set()
    vocab_conut=[]
    vocab_most_e=[]
    for sample in samples:
        r_class=sample['query']
        
        # for @entity
        doc=sample['doc'].strip().split()
        doc= [w.lower() for w in doc if not w.startswith('@entity')]
        #doc= [w.lower() for w in doc if not w.startswith('@entity') and w[0].islower()]
        vocab_most_e.extend(doc)
#        doc=nltk.word_tokenize(sample['doc'].lower().strip())
#        vocab_most_e.extend(doc)
#        
#        e1=sample['e1'].lower()
#        e2=sample['answear'].lower()        
#        doc= [w for w in doc if (w not in e1) and (w not in e2)]
        
        vocab_conut.extend(doc)
        r_set.add(r_class)
    if top_K:
        top_K=top_K
    else:
        top_K=len(list(set(vocab_conut))) #124082

    # not del stop word; pure doc;
    vocab_most_e= [w for w in vocab_most_e if w not in stopwords and w.isalpha()==True]
    most_common_e=[w[0] for w in Counter(vocab_most_e).most_common(choose_top_K)]
    
    vocab=[w[0] for w in Counter(vocab_conut).most_common(top_K) if w[0] not in stopwords] # stopwords
    vocab=list(filter(lambda x:x.isalpha(),vocab))
    #vocab=list(filter(lambda x:not (x.startswith("'") or x.startswith('"') or x.startswith("/") or x.startswith("+") or x.startswith("-") or x.startswith("*") or x.startswith("!")),vocab))

    UNK='UNK'
    vocab.insert(0, UNK)
    vocab_size=len(vocab)
    word_dict=dict(zip(vocab,range(vocab_size)))
    #index_2_word=dict(zip(word_dict.values(),word_dict.keys()))
    
    r_list=list(r_set)
    r_dict=dict(zip(r_list,range(len(r_list))))
    #index_2_r=dict(zip(r_dict.values(),r_dict.keys()))
    
    # write into pickle
    pickle.dump(word_dict,open(w_dict_path,'wb'))
    pickle.dump(r_dict,open(r_dict_path,'wb'))    
    pickle.dump(most_common_e,open(most_e_path,'wb'))  
    return word_dict,r_dict,most_common_e

import numpy as np
def vectorize(sample,word_dict,r_dict,r_id):

     doc=sample['doc'].strip().split()
     doc= [w.lower() for w in doc if not w.startswith('@entity')]
     doc=list(filter(lambda x:x.isalpha(),doc))
     doc= [w for w in doc if w not in stopwords]
     #doc= [w.lower() for w in doc if not w.startswith('@entity') and w[0].islower()]
#     doc=nltk.word_tokenize(sample['doc'].lower().strip())
#
#     e1=sample['e1'].lower()
#     e2=sample['answear'].lower()        
#     doc= [w for w in doc if (w not in e1) and (w not in e2)]
     
     doc=list(map(lambda w:word_dict.get(w,0),doc))
     doc=Counter(doc)
     x=np.zeros(len(word_dict))
     for w in doc:
         x[w]=doc[w]
     x[0]=0 # UNK
     r=sample['query']
     y=1 if r_dict[r]==r_id else 0
     return x,y
