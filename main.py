import csv
import os
from utility import get_good_batches
from Networks import Hidden1 
import torch.nn.functional as func
import torch
from torch import nn
import torch.optim as optim
import time

path=os.path.join(os.getcwd(),"preprocessed")


max_epochs=100

nets=[Hidden1(100),Hidden1(80),Hidden1(1000)]
training=[300,500,700,800,900]
batch_sizes=[100,50,25,20]
#activations=[torch.sigmoid(),torch.relu()]
losses=[nn.CrossEntropyLoss()]
optims=[optim.SGD(nets[1].parameters(), lr=0.1)]
criterion=nn.CrossEntropyLoss()

#Below code is testing a lot of parameters of the networks(architecture, 
#batch size, amount of training samples,activation functions, loss function and
#optimisation of gradient descent)
for n in nets:
    for t in training:
        for b in batch_sizes:
                for l in losses:
                    for o in optims:
                        print("training",t,"batch size",b)
                        #We load the data in batches
                        data_list,class_list, test_set,test_class=get_good_batches(path,b,t)
                        #Let's train the network
                        thisnet=n
                        loss_f=l
                        o=optim.SGD(n.parameters(), lr=0.1)
                        #nb of epochs
                        e=0
                        #If loss doesn't move anymore, we stop training
                        old_loss=10000
                        convergence=False
                        while(e<max_epochs and convergence==False):
                            e+=1
                            for i in range(len(data_list)):
                                batch=data_list[i]
                                classes=class_list[i]
                                outs=torch.zeros(b,10)
                                for k in range(b):
                                    outs[k,:]=n(batch[k,0,:,:])
                                o.zero_grad() 
        
                                loss = criterion(outs,classes)
                                loss.backward()
                                o.step()
                                
                                if abs(loss.item())<0.00001:
                                    print("loss is very small")
                                    convergence=True
                                    break
                                if abs(loss.item()-old_loss)<0.0001:
                                    print("loss does not move")
                                    convergence=True
                                    break
                                old_loss=loss.item()
                            if convergence:
                                print("convergence condition reached after",e,"epochs")
                                break
                        if(convergence==False):
                            print("training done, no convergence")
                        #We will now validate the network
                        errors=0
                        for k in range(1000-t):
                            output=n(test_set[k,0,:,:])
                            predicted_class=output[0,:].max(0)[1]
                            if predicted_class==test_class[k]:
                                pass
                            else:
                                errors+=1
                        print("errors",errors)
                        perc_err=errors/(1000-t)
                        acc=(1-perc_err)*100
                        print("accuracy",acc)
def train_in_full(net,data,classes,activation,loss):
    pass
    