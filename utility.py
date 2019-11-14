#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:28:28 2019

@author: tom
"""
import torch 
import os
from PIL import Image
from torchvision import transforms

def get_repartition(classes):
    rep=torch.zeros(10)
    L=len(classes)
    for c in range(10):
        count=0
        for l in range(L):
            if classes(l)==c:
                count+=1
        rep[c]=count/L
    return rep
        
def get_class(filename):
    return int(filename[7])

def load_data(path):
    files=sorted(os.listdir(path))
    #First we initialize the tensors holding classes and images, knowing that there are
    #1000 10x10 pixels images
    #Because the net will just return us a tensor of probability
    #We need to make  somethng that looks like it
    #We will call it target
    images=torch.zeros(1000,1,10,10)
    target=torch.zeros(1000,10)
    classes=torch.zeros(1000,dtype=torch.long)
    i=0


    for file in files:
        img=Image.open(os.path.join(path,file))
        #We convert the PIL image to a pytorch tensor
        tran=transforms.ToTensor()
        images[i,0,:,:]=tran(img)
        target_slice=torch.zeros(10)
        target_slice[get_class(file)]=1
        target[i,:]=target_slice
        classes[i]=get_class(file)
        i+=1
    return target,classes,images

def get_good_batches(path,batch_nb,train):
    """
    We split the data into n batches with equal repartition of classes to prevent the lossfuntion
    to increase because it discovers another class. This method also should take into account that
    the last batch can be smaller
    """
    reg_batch_size=train//batch_nb
    #We determine the nb of representants of a class there should be per batch
    nb_in_class=reg_batch_size//10
    last_batch_size=0
    class_last_ind=torch.zeros(10)
    """
    Since the samples are in numerical order and there is 100 of it 
    we are sure of the position of the samples
    """
    if train%batch_size!=0:
        reg_batches-=1
        last_batch_size=train%batch_size
        
    for n in range(reg_batches):
        for c in range(10):
            
    
    
    _,classes,images=load_data(path)
    data_list=[]
    class_list=[]
    return data_list,class_list