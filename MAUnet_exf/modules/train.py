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
import numpy as np
np.random.seed(1)
class CustomLoss(nn.Module):
    def __init__(self,weight_factor=2.0):
        super(CustomLoss,self).__init__()
        self.weight_factor=weight_factor
    def forward(self,y_pred,y_true):
        absolute_error=torch.abs(y_pred-y_true)
        threshold=torch.max(absolute_error)*0.6
        # threshold=0.02025
        weighted_error=torch.where(absolute_error<threshold,absolute_error,self.weight_factor*absolute_error)
        loss=torch.mean(weighted_error)
        return loss
class QuantileLoss(nn.Module):
    def __init__(self,ql,qr):
        super(QuantileLoss,self).__init__()
        self.ql=ql
        self.qr=qr
    def forward(self,y_pred,y_true):
        errors=y_true-y_pred
        loss=torch.max((self.ql-1)*errors,self.qr*errors)
        return loss.mean()
class QuantileLoss_qr(nn.Module):
    def __init__(self,qr):
        super(QuantileLoss_qr,self).__init__()
        self.quantile=qr
    def forward(self,y_pred,y_true):
        errors=y_true-y_pred
        loss=torch.max((self.quantile-1)*errors,self.quantile*errors)
        return loss.mean()
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

def compute_sae(targets,predictions):
    diff=targets-predictions
    square=diff**2
    total_diff=torch.sum(square)
    sae=total_diff/targets.numel()
    return sae

def calculate_f1(targets,predictions,device):
    tp=sum([1 for t,p in zip(targets,predictions) if t==1 and p==1])
    tp=tp
    fp=sum([1 for t,p in zip(targets,predictions) if t==0 and p==1])
    fp=fp
    fn=sum([1 for t,p in zip(targets,predictions) if t==1 and p==0])
    fn=fn
    if tp + fp == 0 or tp + fn==0:
        f1_score = 0
    else:
        precision=tp/(tp+fp)
        precision=precision
        recall=tp/(tp+fn)
        recall=recall
        if precision ==0 or recall ==0:
            f1_score=0
        else:
            f1_score=2*(precision*recall)/(precision+recall)
    return f1_score
def calculate_f1_score(targets,predictions,device):
    maxdrop = torch.max(targets)
    maxdrop=maxdrop.to(device=device,dtype=torch.float32)
    threshold = 0.6 * maxdrop
    threshold=threshold.to(device=device,dtype=torch.float32)
    # tensor1 = targets.view(-1)
    # tensor2 = predictions.view(-1)
    tensor1 = targets.to(device=device,dtype=torch.float32)
    tensor2 = predictions.to(device=device,dtype=torch.float32)
    ten1=torch.tensor(1.)
    ten1=ten1.to(device=device,dtype=torch.float32)
    ten0= torch.tensor(0.)
    ten0=ten0.to(device=device,dtype=torch.float32)
    pred = torch.where(tensor2 >= threshold,ten1, ten0)
    labels = torch.where(tensor1 >= threshold, ten1, ten0)
    f1 = calculate_f1(pred, labels,device)
    return f1

def calculate_f1_loss(targets,predictions,device):
    maxdrop = torch.max(targets)
    maxdrop=maxdrop.to(device=device,dtype=torch.float32)
    threshold = 0.8 * maxdrop
    threshold=threshold.to(device=device,dtype=torch.float32)
    # tensor1 = targets.view(-1)
    # tensor2 = predictions.view(-1)
    tensor1 = targets.to(device=device,dtype=torch.float32)
    tensor2 = predictions.to(device=device,dtype=torch.float32)
    indices=torch.where(tensor1>threshold)
    loss=torch.mean((tensor1[indices]-tensor2[indices])**2)
    return loss


def train_net(model,device,data_list,tdata_list,epochs,lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=50)
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
   #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # criterion = QuantileLoss_qr(qr=0.6)

    # criterion=CustomLoss(2)
    for epoch in range(epochs):
        model.train()
        tloss=0
        tf1=0
        tmae=0
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

            mae=compute_mae(pred,y)
            mse=mae
            trainloss=mse
            trainloss.backward()
            optimizer.step()
            tloss+=trainloss.item()
        scheduler.step()
        print(f'epoch {epoch} loss:{tloss} mae:{tmae} f1:{tf1}')
    print("training finished")
    return model



