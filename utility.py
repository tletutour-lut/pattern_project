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
    L=classes.size()[0]
    for c in range(10):
        count=0
        for l in range(L):
            if classes[l]==c:
                count+=1
        rep[c]=count/L*100
    return rep
        
def get_class(filename):
    return int(filename[7])

def load_data(path,sort=True):
    if sort:
        files=sorted(os.listdir(path))
    else:
        files=os.listdir(path)
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

def get_good_batches(path,batch_size,train):
    """
    We split the data into n batches with equal repartition of classes to prevent the lossfuntion
    to increase because it discovers another class.
    TODO This method also should take into account that
    the last batch can be smaller
    """
    batch_nb=train//batch_size
    #We determine the nb of representants of a class there should be per batch
    nb_in_class=batch_size//10
    class_last_ind=torch.zeros(10)
    _,classes,images=load_data(path)
    data_list=[]
    class_list=[]
    """
    Since the samples are in numerical order and there is 100 of it 
    we are sure of the class of the position of the samples
    """
    for n in range(batch_nb):
        print("batch nb",n)
        batch=torch.zeros(batch_size,1,10,10)
        batch_class=torch.zeros(batch_size,dtype=torch.long)
        k=0
        for c in range(10):
            print("class",c)
            for b in range(nb_in_class):
                
                index=100*c+int(class_last_ind[c])
                batch[k,:,:,:]=images[index,:,:,:]
                batch_class[k]=classes[index]
                class_last_ind[c]+=1
                k+=1
        data_list.append(batch)
        class_list.append(batch_class)
        print(get_repartition(batch_class))
        
    """
    We also define the test set
    """
    test=1000-train
    nb_in_test=test//10
    test_set=torch.zeros(test,1,10,10)
    test_class=torch.zeros(test,dtype=torch.long)
    k=0
    for c in range(10):
        for b in range(nb_in_test):
            index=100*c+int(class_last_ind[c])
            test_set[k,:,:,:]=images[index,:,:,:]
            test_class[k]=classes[index]
            class_last_ind[c]+=1
            k+=1
    
    return data_list,class_list, test_set,test_class