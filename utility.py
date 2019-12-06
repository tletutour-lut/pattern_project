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
import matplotlib as plt
import torch.optim as optim

def kfold_lists(path,K):
    """
    this method spits out data, classes and target lists for to implement cross
    validation, unfortunately I don't have time to make it with equal class repartition
    loss will be bumpy, but cross validation is important
    1000 has to be a multiple of K, it could work without but no time for this
    """
    
    target,classes,images=load_data(path,sort=False)
    if(1000%K!=0):
        exit("1000 has to be a multiple of K")
    fold_size=1000//K
    target_list=[]
    classes_list=[]
    images_list=[]
    for n in range(K):
        target_list.append(target[n:n+fold_size,:])
        classes_list.append(classes[n:n+fold_size])
        images_list.append(images[n:n+fold_size,:,:,:])
        
    return target_list,classes_list, images_list

def split_in_batches(target,classes, images,batch_size):
    N=classes.shape[0]
    batch_nb=classes.shape[0]//batch_size
    #We determine the nb of representants of a class there should be per batch
    nb_in_class=batch_size//10
    class_last_ind=torch.zeros(10)
    image_list=[]
    target_list=[]
    class_list=[]
    k=0
    #We assume that the data is already ordered, which it will be because we 
    #call this method after the ordered folds
    for n in range(batch_nb):
        batch_target=torch.zeros(batch_size,10)
        batch_image=torch.zeros(batch_size,1,8,8)
        batch_class=torch.zeros(batch_size,dtype=torch.long)
        k=0
        for c in range(10):
            for b in range(nb_in_class):
                index=(N//10)*c+int(class_last_ind[c])
                batch_target[k,:]=target[index,:]
                batch_image[k,:,:,:]=images[index,:,:,:]
                batch_class[k]=classes[index]
                class_last_ind[c]+=1
                k+=1
            #print("class last ind=",class_last_ind[c])
        image_list.append(batch_image)
        class_list.append(batch_class)
        target_list.append(batch_target)
        #print(get_repartition(batch_class))
    return target_list,class_list,image_list

def get_repartition(classes):
    """
    I made this method when I noticed that the loss was "bumpy", and wanted to 
    test the hypothesis that due to the randomness of the class distribution
    it is just a testing method, doesn't do anything for the training
    """
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
    """
    Loads the data, target, and classes from the PNG images
    sorted means that the data will be in class order, ie. 00000....1111 etc
    """
    if sort:
        files=sorted(os.listdir(path))
    else:
        files=os.listdir(path)
    #First we initialize the tensors holding classes and images, knowing that there are
    #1000 10x10 pixels images
    #Because the net will just return us a tensor of probability
    #We need to make  somethng that looks like it
    #We will call it target
    images=torch.zeros(1000,1,8,8)
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


def get_good_folds(path,K):
    """
    We split the data into n batches with equal repartition of classes to prevent the lossfuntion
    to increase because it discovers another class.
    TODO This method also should take into account that
    the last batch can be smaller
    """
    fold_size=1000//K
    #We determine the nb of representants of a class there should be per batch
    nb_in_class=fold_size//10
    class_last_ind=torch.zeros(10)
    target,classes,images=load_data(path)
    target_list=[]
    data_list=[]
    class_list=[]
    """
    Since the samples are in numerical order and there is 100 of it 
    we are sure of the class of the position of the samples
    """
    for n in range(K):
        fold_image=torch.zeros(fold_size,1,8,8)
        fold_class=torch.zeros(fold_size,dtype=torch.long)
        fold_target=torch.zeros(fold_size,10)
        j=0
        for c in range(10):
            for b in range(nb_in_class):
                index=100*c+int(class_last_ind[c])
                fold_image[j,:,:,:]=images[index,:,:,:]
                fold_target[j,:]=target[index,:]
                fold_class[j]=classes[index]
                class_last_ind[c]+=1
                j+=1

        data_list.append(fold_image)
        target_list.append(fold_target)
        class_list.append(fold_class)
    
    return target_list, class_list,data_list

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
        batch=torch.zeros(batch_size,1,8,8)
        batch_class=torch.zeros(batch_size,dtype=torch.long)
        k=0
        for c in range(10):
            for b in range(nb_in_class):
                index=100*c+int(class_last_ind[c])
                batch[k,:,:,:]=images[index,:,:,:]
                batch_class[k]=classes[index]
                class_last_ind[c]+=1
                k+=1
            #print("class last ind=",class_last_ind[c])
        data_list.append(batch)
        class_list.append(batch_class)
        #print(get_repartition(batch_class))
        
    """
    We also define the test set
    """
    test=1000-train
    nb_in_test=test//10
    test_set=torch.zeros(test,1,8,8)
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