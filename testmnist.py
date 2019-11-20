#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:45:35 2019

@author: tom
"""
import torch
import os
import torchvision
from PIL import Image
from torch import nn
import torch.nn.functional as func
import matplotlib as plt
from torchvision import transforms 
import torch.optim as optim

data_path="/home/tom/Documents/LUT/Pattern_Recog/digits_3d_training_data/digits_3d/training_data/preprocessed/"
def get_class(filename):
    return int(filename[7])

def get_class_rep(imin,imax):
    """
    This method accounts for how well all the classes are represented 
    from index imin imad
    Result is a torch array containing percentages of class representations
    suspected fucking up cause of loss func
    """
    res=torch.zeros(10)
    nb=imax-imin
    
    if(nb<=0):
        exit("imax should be bigger than imin")
        
    _,classes,_=load_data(data_path)
    for i in range(10):
        counted=0
        for k in range(imin,imax):
            if classes[k]==i:
                counted+=1        
            res[i]=counted/nb*100
    return res

def load_data(path):
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


class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        #self.conv1=nn.Conv2d(1,6,5)
        #self.conv2=nn.Conv2d(6,16,5) 
        self.input=nn.Linear(10*10,200)
        self.hidden1=nn.Linear(200,200)
        self.hidden2=nn.Linear(200,200)
        self.hidden3=nn.Linear(200,200)
        self.output=nn.Linear(200,10)

        
    def forward(self,x):
        #x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        #x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x=x.view(-1,10*10)
        x=func.relu(self.input(x))
        x=func.relu(self.hidden1(x))
        x=func.relu(self.hidden2(x))
        x=func.relu(self.hidden3(x))
        x=self.output(x)
        return x
    

def get_out_shape(x):
    size = x.size()[1:]
    res=1
    for s in size:
        res*=s
    #print(res)
    return res


def main():
    
    #We get the data from file using the above declared method
    target, classes,images=load_data(data_path)
    
    #First lets differentiate the training and testing data
    #Let's separate the training data into mini batches of 50
    train=900
    epochs=20
    batch_size=300
    #We initialize the net as well as the loss criterion & the opti 
    net=Network()
    criterion= nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.7)
    
    
    
    if(train%batch_size!=0):
        exit("train should be a perfect multiple of batch_size")
    nb_batch=train//batch_size
    batch_list=[]
    class_list=[]
    for i in range(nb_batch):
        batch_list.append(images[i*batch_size:(i+1)*batch_size,:,:,:])
        class_list.append(classes[i*batch_size:(i+1)*batch_size])
    if(train%batch_size!=0):
        exit("train should be a perfect multiple of batch_size")
    nb_batch=train//batch_size
    for e in range(epochs):
    
        for i in range(len(batch_list)):
            batch=batch_list[i]
            classes=class_list[i]
            out_classes=torch.zeros(batch_size)
            outs=torch.zeros(batch_size,10)
            for k in range(batch_size):
                out=net(batch[k,0,:,:])
                outs[k]=out
                out_classes[k]=out[0,:].max(0)[1]
            
            optimizer.zero_grad() 
    
            loss = criterion(outs,classes)
            loss.backward(retain_graph=True)
            optimizer.step()
            print("epoch",e+1,"batch",i+1,"loss",loss)
    
    #We can now test our network
    outtest=torch.zeros(1000-train,10)
    errors=0          
    for k in range(train,1000):
            outtest[k-train,:]=net(images[k,:,:])
            #Now we have to compare with the actual truth
            #Essentially if max of outtest[k-train,:]is at the same index
            #than max target(k,:) classification is correct
            #else error+1
            if outtest[k-train,:].max(0)[1]==target[k,:].max(0)[1]:
                #print("hueeeee")
                pass
            else:
                errors+=1
    print("errors=",errors)
    #Let us define a measure of accuracy
    #First the percentage of error amongst the test set
    perc_err=errors/(1000-train)
    acc=(1-perc_err)*100
    print("accuracy :",acc,"%")
