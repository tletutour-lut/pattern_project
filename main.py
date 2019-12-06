import csv
import os
from utility import get_good_batches,get_good_folds,split_in_batches
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
                        #We load the data in folds representing classes equally
                        K=10
                        target_fold, class_fold,image_fold=get_good_folds(path,K)
                        for k in range(K):
                            print("validating on fold ",k+1,"training on all the others")
                            #We reinstantiate the network for each of the tested
                            #folds otherwise the networks trains on data it already saw
                            thisnet=n
                            loss_f=l
                            o=optim.SGD(n.parameters(), lr=0.1)
                            for kt in range(K):
                                if k!=kt:
                                    #If the fold isnt the validation one
                                    #We train the network
                                    target_list,class_list,image_list=split_in_batches(target_fold[kt],class_fold[kt], image_fold[kt],b)
                                    #nb of epochs
                                    e=0
                                    #If loss doesn't move anymore, we stop training
                                    old_loss=10000
                                    convergence=False
                                    
                                    while(e<max_epochs and convergence==False):
                                        e+=1
                                        for i in range(len(class_list)):
                                            batch=image_list[i]
                                            classes=class_list[i]
                                            outs=torch.zeros(b,10)
                                            for z in range(classes.shape[0]):
                                                outs[z,:]=n(batch[z,0,:,:])
                                            o.zero_grad() 
                    
                                            loss = criterion(outs,classes)
                                            loss.backward()
                                            o.step()
                                            
                                            if abs(loss.item())<0.00001:
                                                #print("loss is very small")
                                                convergence=True
                                                break
                                            if abs(loss.item()-old_loss)<0.0001:
                                                #print("loss does not move")
                                                convergence=True
                                                break
                                            old_loss=loss.item()
                                        if convergence:
                                            #print("convergence condition reached after",e,"epochs")
                                            break
                            if(convergence==False):
                                #print("training done, no convergence")
                                pass
                            #We will now validate the network on fold k
                            errors=0
                            test_set=image_fold[k]
                            test_class=class_fold[k]
                            for idx in range(test_class.shape[0]):
                                output=n(test_set[idx,0,:,:])
                                predicted_class=output[0,:].max(0)[1]
                                if predicted_class==test_class[idx].type(torch.LongTensor):
                                    pass
                                else:
                                    errors+=1
                            print("errors",errors)
                            perc_err=errors/(1000//K)
                            acc=(1-perc_err)*100
                            print("accuracy",acc)
def train_in_full(net,data,classes,activation,loss):
    pass
    