#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:28:19 2019

@author: tom
"""
from Networks import Lin3Net,Conv2Lin3
import torch.optim as optim
from utility import load_data
from torch import nn
from sklearn.metrics import confusion_matrix
import torch

data_path="/home/tom/Documents/LUT/project/pattern_project/preprocessed/"
#We get the data from file using the above declared method
target, classes,images=load_data(data_path)

#First lets differentiate the training and testing data
#Let's separate the training data into mini batches of 50
train=900
epochs=50
batch_size=300
#We initialize the net as well as the loss criterion & the opti 
net=Lin3Net()
criterion= nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.7)
if(train%batch_size!=0):
    exit("train should be a perfect multiple of batch_size")
    nb_batch=train//batch_size
batch_list=[]
class_list=[]
nb_batch=train//batch_size
for i in range(nb_batch):
    batch_list.append(images[i*batch_size:(i+1)*batch_size,:,:,:])
    class_list.append(classes[i*batch_size:(i+1)*batch_size])


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
guesses=torch.zeros(1000-train)
errors=0          
for k in range(train,1000):
        outtest[k-train,:]=net(images[k,:,:])
        #Now we have to compare with the actual truth
        #Essentially if max of outtest[k-train,:]is at the same index
        #than max target(k,:) classification is correct
        #else error+1
        guesses[k-train]=outtest[k-train,:].max(0)[1]
        if outtest[k-train,:].max(0)[1]==target[k,:].max(0)[1]:
            #print("hueeeee")
            pass
        else:
            errors+=1
print("errors=",errors)
print(classes[train:1000].size())
print(confusion_matrix(classes[train:1000],guesses))
#Let us define a measure of accuracy
#First the percentage of error amongst the test set
perc_err=errors/(1000-train)
acc=(1-perc_err)*100
print("accuracy :",acc,"%")
