#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:26:22 2019

@author: tom
"""

import torch
from torch import nn
import torch.nn.functional as func





class NetworkTest(nn.Module):
    
    def __init__(self):
        super(NetworkTest, self).__init__()
        #self.conv1=nn.Conv2d(1,6,5)
        #self.conv2=nn.Conv2d(6,16,5) 
        self.input=nn.Linear(8*8,300)
        self.hidden1=nn.Linear(300,300)
        self.hidden2=nn.Linear(300,300)
        self.hidden3=nn.Linear(300,300)
        self.hidden4=nn.Linear(300,300)
        self.hidden5=nn.Linear(300,300)
        self.output=nn.Linear(300,10)

        
    def forward(self,x):
        #x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        #x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x=x.view(-1,8*8)
        x=func.relu(self.input(x))
        x=func.relu(self.hidden1(x))
        x=func.relu(self.hidden2(x))
        x=func.relu(self.hidden3(x))
        x=func.relu(self.hidden4(x))
        x=func.relu(self.hidden5(x))
        x=self.output(x)
        return x
    
class Hidden1(nn.Module):
    def __init__(self,h,a):
        super(Hidden1, self).__init__() 
        self.hidden=nn.Linear(8*8,h)
        self.output=nn.Linear(h,10)
        self.a=a
        self.h=h
    def forward(self,x):
        x=x.view(-1,8*8)
        if self.a=='relu':   
            x=torch.relu(self.hidden(x))
        if self.a=='sigmoid':   
            x=torch.sigmoid(self.hidden(x))
        return self.output(x)
    
    
class Lin4Net(nn.Module):
    
    def __init__(self,h,a):
        super(Lin4Net, self).__init__() 
        self.fc1=nn.Linear(8*8,h)
        self.fc2=nn.Linear(h,h)
        self.fc3=nn.Linear(h,h)
        self.fc4=nn.Linear(h,10)
        self.a=a
        self.h=h
        
    def forward(self,x):
        #x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        #x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x=x.view(-1,8*8)
        if self.a=='relu':   
            x=torch.relu(self.fc1(x))
            x=torch.relu(self.fc2(x))
            x=torch.relu(self.fc3(x))
        if self.a=='sigmoid':   
            x=torch.sigmoid(self.fc1(x))
            x=torch.sigmoid(self.fc2(x))
            x=torch.sigmoid(self.fc3(x))
        x=self.fc4(x)
        return x
    
class Conv2Lin3(nn.Module):
    def __init__(self):
        super(Conv2Lin3, self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(10*10,10)
    def forward(self,x):
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        print(get_size_mult(x))
        x=x.view(-1,get_size_mult(x))
        
        
def get_size_mult(x):
    res=1
    """
    this is after passing by the max pooling layer
    this is to pass the array to a flat version 
    """
    for dim in x.dim():
        res=res*torch.size()[dim]