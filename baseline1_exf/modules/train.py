import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split

class PixelwiseLoss(nn.Module):
    def forward(self,inputs,targets):
        return F.smooth_l1_loss(inputs,targets)
def l2_reg_loss(model,l2_alpha):
    l2_loss=[]
    for model in model.modules():
        if type(model) is nn.Conv2d:
            l2_loss.append((model.weight ** 2).sum()/2.0)
    return l2_alpha*sum(l2_loss)
def compute_mae(targets,predictions):
    diff=targets-predictions
    abs_diff=torch.abs(diff)
    total_diff=torch.sum(abs_diff)
    mae=total_diff/targets.numel()
    return mae
def compute_mse(targets,predictions):
    diff=(targets-predictions)**2
    total_diff=torch.sum(diff)
    mse=total_diff/targets.numel()
    return mse
def train_net(model,device,data_list,tdata_list,epochs,lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=50)
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
   #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # criterion = PixelwiseLoss()
    for epoch in range(epochs):
        model.train()
        tloss=0
        for x,y in zip(data_list,tdata_list):
            optimizer.zero_grad()
            x=x.to(device=device,dtype=torch.float32)
            y=y.to(device=device,dtype=torch.float32)
            pred=model(x)
            for d in range(1):
                ty=torch.squeeze(pred[d,:,:])
                max=model.ymax[d]
                min=model.ymin[d]
                normal=ty*(max-min)+min
                pred[d,:,:]=normal
            trainloss=compute_mae(y,pred)
            trainloss.backward()
            optimizer.step()
            scheduler.step()
            tloss+=trainloss.item()
        print(f'epoch {epoch} loss:{tloss}')
    print("training finished")
    return model

def train_net_asap7(model,device,data_list,tdata_list,epochs,lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=50)
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
   #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # criterion = PixelwiseLoss()
    for epoch in range(epochs):
        model.train()
        tloss=0
        count=0
        for x,y in zip(data_list,tdata_list):
            optimizer.zero_grad()
            x=x.to(device=device,dtype=torch.float32)
            y=y.to(device=device,dtype=torch.float32)
            pred=model(x)
            for d in range(1):
                ty=torch.squeeze(pred[d,:,:])
                max=model.ymax[d]
                min=model.ymin[d]
                normal=ty*(max-min)+min
                pred[d,:,:]=normal
            # y=F.interpolate(y,size=(pred.size(2),pred.size(3)),mode="bilinear")
            if y.size(2) != pred.size(2):
                count+=1
                print(x.shape,y.shape,pred.shape)
            else:
                trainloss=compute_mae(y,pred)
                trainloss.backward()
                optimizer.step()
                scheduler.step()
                tloss+=trainloss.item()
        print(f'epoch {epoch} loss:{tloss} {count}')
    print("training finished")
    return model