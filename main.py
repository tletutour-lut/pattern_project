import csv
import os
from utility import get_good_folds,split_in_batches
from Networks import Hidden1,Lin4Net
import torch.nn.functional as func
import torch
from torch import nn
import torch.optim as optim
import time

path=os.path.join(os.getcwd(),"preprocessed")


max_epochs=100

nets=[Hidden1(100,'sigmoid'),
      Hidden1(100,'relu'),
      Hidden1(80,'relu'),
      Hidden1(80,'sigmoid'),
      Hidden1(1000,'relu'),
      Hidden1(1000,'sigmoid'),
      Lin4Net(100,'sigmoid'),
      Lin4Net(100,'relu'),
      Lin4Net(80,'relu'),
      Lin4Net(80,'sigmoid'),
      Lin4Net(1000,'relu'),
      Lin4Net(1000,'sigmoid')]


batch_sizes=[100,50,25,20]
losses=[nn.L1Loss(),nn.CrossEntropyLoss()]
optims=['sgd','adam']
criterion=nn.CrossEntropyLoss()

#Below code is testing a lot of parameters of the networks(architecture, 
#batch size, amount of training samples,activation functions, loss function and
#optimisation of gradient descent)
with open(str(int(time.time()))+".csv",'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["net type","activation","nb of hidden neuron per layer","batch_size :","loss function :","optim","average accuracy","average training time"])
    for n in nets:
        for l in losses:
            for o_str in optims:
                for b in batch_sizes:
            
                
                    print("net type","batch_size :",b,"loss function :",l,"optim",o_str)
                    #We load the data in folds representing classes equally
                    K=10
                    target_fold, class_fold,image_fold=get_good_folds(path,K)
                    sum_acc=0
                    sum_time=0
                    for k in range(K):
                        t_start=time.time()
                        print("validating on fold ",k+1,"training on all the others")
                        #We reinstantiate the network for each of the tested
                        #folds otherwise the networks trains on data it already saw
                        thisnet=n
                        loss_f=l
                        o=None
                        if o_str=="sgd":
                            o=optim.SGD(n.parameters(), lr=0.1)
                        if o_str=="adam":
                            o=optim.Adam(n.parameters(), lr=0.1)
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
                                        if abs(loss.item()-old_loss)<0.001:
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
                        sum_time+=time.time()-t_start
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
                        print("\terrors",errors)
                        perc_err=errors/(1000//K)
                        acc=(1-perc_err)*100
                        sum_acc+=acc
                        print("\taccuracy",acc)
                    average_time=sum_time/K
                    average_acc=sum_acc/K
                    writer.writerow([type(n),n.a,n.h,b,l,o_str,average_acc,average_time])
def train_in_full(net,data,classes,activation,loss):
    pass
    