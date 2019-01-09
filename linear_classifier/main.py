#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:30:37 2018

@author: xuweijia
"""
# 为每一个关系，训练一个二分类器y=sigmoid(x*theta)　　theta每个单词对分类的贡献. 该值越大，该特征越好，该位置对应单词越能反映关系r
# 每个样本，类型r_id
# 输入样本：x: １，Ｖ　　　是tf向量. 句子中每个单词在该句子中出现的次数
#           y: 0/1 
# 所有样本都可以利用上.test  acc,f1
#  预测： theta: 最大的theta对应位置的单词    
import torch
from expand_words import vectorize,make_dict,to_vars,evaluate
from loader import Data
from model import Logisit_model
import numpy as np
import json
import os
import time
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s',datefmt='%m-%d %H:%M')
from torch.nn.utils import clip_grad_norm

batch_size=128
Epoch=10
LEARNING_RATE=0.01
USE_CUDA=True
GRAD_CLIP=5
print_every=200
Beta=0.5  # >1, more concern recall; should concern precision
top_K_word=10  # save top K words for every relation
print_dev=2000

#top_K=100000
choose_top_K=15000 # when pick r's top words,choose most common words
 
# data
father_file='90w'
train_data_file='train_90w_list.json'
dev_data_file='dev_90w_list.json'
test_data_file='test_90w_list.json'

# p
p_train_dict='p_dict/p_dict_train'
p_dev_dict='p_dict/p_dict_dev'
p_test_dict='p_dict/p_dict_test'

dict_file='word_dict.pkl'
r_dict_file='r_dict.pkl'
most_e_file='most_e.pkl'
data_file=[train_data_file,dev_data_file,test_data_file]

# p
with open(p_train_dict,'r') as f:
    train_dict=json.load(f)
with open(p_dev_dict,'r') as f:
    dev_dict=json.load(f)
with open(p_test_dict,'r') as f:
    test_dict=json.load(f)
# sort p 
high_p=sorted(train_dict.items(), key=lambda item :item[1],reverse=True)
word_dict,_,most_common_e=make_dict(father_file,data_file,dict_file,r_dict_file,most_e_file,choose_top_K,top_K=None)

r_dict=dict(zip([w[0] for w in high_p],range(len(high_p))))

index_2_word=dict(zip(word_dict.values(),word_dict.keys()))
index_2_r=dict(zip(r_dict.values(),r_dict.keys()))

r_words={}

with open(os.path.join(father_file,train_data_file),'r') as f:
    train_samples=json.load(f)

with open(os.path.join(father_file,dev_data_file),'r') as f:
    dev_samples=json.load(f)
    
with open(os.path.join(father_file,test_data_file),'r') as f:
    test_samples=json.load(f)

save_path = ('experiments_{}/'.format(choose_top_K))
if not os.path.exists(save_path): os.makedirs(save_path)


# different model
# def train_one_model(r_id)

# parellel

start = time.time()
for model_id in range(len(r_dict)):
    # one model
    words=0
    r=index_2_r[model_id]
    # only train r in dev
    if r not in dev_dict:
        continue
    model=Logisit_model(len(word_dict))
    # opt = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=LEARNING_RATE)  
    # sgd
    # opt= torch.optim.RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, alpha=0.9)
    opt=torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9)
    if USE_CUDA:
        model.cuda()
        
    # load existing best_model
    if os.path.isfile(os.path.join(save_path,'{}_{}_best_model.pkl'.format(model_id,r))):
        print('loading previously best model id{} {}'.format(model_id,r))
        model.load_state_dict(torch.load(os.path.join(save_path,'{}_{}_best_model.pkl'.format(model_id,r))))
    # load existing train_model
    elif os.path.isfile(os.path.join(save_path,'{}_{}_init_model.pkl'.format(model_id,r))):
        print('loading init model id{} {}'.format(model_id,r))
        model.load_state_dict(torch.load(os.path.join(save_path,'{}_{}_init_model.pkl'.format(model_id,r))))        
        
    # loader
    train_dataset=Data(train_samples,model_id,word_dict,r_dict)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
    dev_dataset=Data(dev_samples,model_id,word_dict,r_dict)
    dev_data_loader = torch.utils.data.DataLoader(dev_dataset,batch_size,shuffle=False)    
    test_dataset=Data(test_samples,model_id,word_dict,r_dict)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size,shuffle=False)   
    
    best_valid_acc = best_test_acc = 0
    best_valid_f1 = best_test_f1=0
    best_valid_fb = best_test_fb=0
    print('-' * 50)
    print("Start training ...")
    for epoch in range(Epoch):
        new_max=False
        if epoch >= 2:
            for param_group in opt.param_groups:
                param_group['lr'] /= 2
# train
        model.train()
        loss = n_examples = acc = TP = n_true = n_predict_true = 0
        for it,ex in enumerate(train_data_loader):
            x,y=ex                                               # x: torch.DoubleTensor of size BxV   y:torch.LongTensor of size B
            n_examples+=x.size()[0]
            # to Variable
            x,y=to_vars([x,y],use_cuda=USE_CUDA,evaluate=False)   # Variable
            
            loss_b,acc_b,TP_b,n_true_b,n_predict_true_b= model(x,y) # tensor.float size 1

            loss += loss_b.cpu().data.numpy()[0] # numpy [1]
            
            acc  += acc_b  # correct number
            TP   += TP_b   # true positive
            n_predict_true+= n_predict_true_b # predict  pos
            n_true+= n_true_b                 # all real pos
            
            opt.zero_grad()
            loss_b.backward()
            clip_grad_norm(parameters=filter(lambda p: p.requires_grad, model.parameters()),max_norm=GRAD_CLIP)
            opt.step()
            if it % print_every == 0 and it!=0:
                spend = (time.time() - start) / 3600
                statement = "r_id :{}, r:{}, Epoch: {}, it: {} (max: {}), "\
                    .format(model_id,r,epoch, it, len(train_data_loader))

                if TP!=0 and n_true!=0 and n_predict_true!=0:
                    precision = 1.0 * TP / n_predict_true          # all predict
                    recall = 1.0 * TP / n_true                     # all truth, have how many
                    f1 = ( 2.0 * precision * recall) / (precision + recall)
                    fb = ((1.0+ Beta**2) * precision * recall) / (Beta**2 * precision + recall)
                    statement += "loss: {:.3f}, acc: {:.3f}, f1: {:.3f},fb: {:.3f},time: {:.1f}(h)"\
                        .format(loss / print_every, acc / n_examples,f1, fb,spend)
                    print(statement)
                else:
                    statement += "loss: {:.3f}, acc: {:.3f},time: {:.1f}(h)"\
                        .format(loss / print_every, acc / n_examples,spend)
                    print(statement)
                loss = n_examples = acc = TP = n_true = n_predict_true =0
                 
                torch.save(model.state_dict(),os.path.join(save_path,'{}_{}_init_model.pkl'.format(model_id,r)))
# dev
            if it % print_dev == 0 and it!=0:
                model.eval()
                
                dev_loss,dev_acc,dev_f1,dev_fb=evaluate(model, dev_data_loader, USE_CUDA,Beta)
                spend = (time.time() - start) / 3600
                statement = "Valid loss: {:.3f}, acc: {:.3f}, f1: {:.3f},fb: {:.3f},time: {:.1f}(h)"\
                    .format(dev_loss, dev_acc, dev_f1,dev_fb,spend)
                print(statement)
                if best_valid_f1 < dev_f1: #and best_valid_acc <= dev_acc:
                    best_valid_f1 = dev_f1
                    if best_valid_acc<dev_acc:
                        best_valid_acc=dev_acc
                    if best_valid_fb < dev_fb:
                        best_valid_fb = dev_fb 
                    #best_valid_acc = dev_acc
                    new_max=True
                    # store best valid model
                    torch.save(model.state_dict(),os.path.join(save_path,'{}_{}_best_model.pkl'.format(model_id,r)))
                    
                    Theta=list(model.parameters())[0].cpu().data.numpy()[0]
                    # index=np.argpartition(-Theta,top_K_word)[:top_K_word]
                    #index=np.argsort(-Theta)[:top_K_word]
                    index=np.argsort(-Theta)
                    words= list(map(lambda i:index_2_word[i],index))
                    words=[w for w in words if w in most_common_e]
                    words=words[:top_K_word]
                    r_words[r]=words
                    #torch.save(model,os.path.join(save_path,'best_model.pkl'))
                print("Best valid acc: {:.3f} f1: {:.3f},fb: {:.3f}".format(best_valid_acc,best_valid_f1,best_valid_fb))
                #print("Best valid f1: {:.3f},fb: {:.3f}".format(best_valid_f1,best_valid_fb))
                print("r_id :{},r :{}, words:{}".format(model_id,r,str(words)))
                model.train()
#-------test-------#
        model.eval()
        test_loss,test_acc,test_f1,test_fb=evaluate(model, test_data_loader, USE_CUDA,Beta)
        spend = (time.time() - start) / 3600
        print("Test loss: {:.3f}, acc: {:.3f}, f1: {:.3f},fb: {:.3f},time: {:.1f}(m)"\
                     .format(test_loss, test_acc, test_f1,test_fb,spend))
        if best_test_f1 < test_f1:
            best_test_f1 = test_f1
            if best_test_acc<test_acc:
                best_test_acc=test_acc
            if best_test_fb < test_fb:
                best_test_fb = test_fb 
        print("Best test acc: {:.3f} f1: {:.3f},fb: {:.3f}".format(best_test_acc,best_test_f1,best_test_fb))
        print("r_id :{},r :{}, words:{}".format(model_id,r,str(words)))
        # just use to check
        if not new_max and epoch >= 3: # until epoch no new accuracy
            break
    if words==0:
        Theta=list(model.parameters())[0].cpu().data.numpy()[0]
        index=np.argsort(-Theta)
        words= list(map(lambda i:index_2_word[i],index))
        words=[w for w in words if w in most_common_e]
        words=words[:top_K_word]
        r_words[r]=words
    with open('expand_words_list','a') as f:
        f.write(r+str(words)+'\n')




with open('expand_words_90wTrue_{}_sgd'.format(choose_top_K),'w') as f:
    json.dump(r_words,f)



