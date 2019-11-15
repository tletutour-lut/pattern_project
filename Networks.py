#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:26:22 2019

@author: tom
"""

import torch
from torch import nn
import torch.nn.functional as func



class Lin1Net(nn.Module):
    def __init__(self):
        super(Lin1Net, self).__init__() 
        self.fc1=nn.Linear(10*10,10)
    def forward(self,x):
        x=x.view(-1,10*10)   
        return torch.sigmoid(x)


class Lin4Net(nn.Module):
    
    def __init__(self):
        super(Lin4Net, self).__init__() 
        self.fc1=nn.Linear(10*10,200)
        self.fc2=nn.Linear(200,200)
        self.fc3=nn.Linear(200,100)
        self.fc4=nn.Linear(100,10)

        
    def forward(self,x):
        #x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2))
        #x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x=x.view(-1,10*10)
        x=func.relu(self.fc1(x))
        x=func.relu(self.fc2(x))
        x=func.relu(self.fc3(x))
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