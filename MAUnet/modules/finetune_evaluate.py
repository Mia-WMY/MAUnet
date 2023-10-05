import os.path
import torch
import numpy as np
from torch import optim
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from process import read_fake_data, read_real_data
from normalization import getxStat,getyStat
from skimage.metrics import structural_similarity as ssim
class PixelwiseLoss(nn.Module):
    def forward(self,inputs,targets):
        return F.smooth_l1_loss(inputs,targets)
def l2_reg_loss(model,l2_alpha):
    l2_loss=[]
    for model in model.modules():
        if type(model) is nn.Conv2d:
            l2_loss.append((model.weight ** 2).sum()/2.0)
    return l2_alpha*sum(l2_loss)
def r2_score(targets, predictions):
    mean_targets = torch.mean(targets)
    ss_total = torch.sum((targets - mean_targets)**2)
    ss_residual = torch.sum((targets - predictions)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item()
def calculate_f1_score(targets,predictions):
    tp=sum([1 for t,p in zip(targets,predictions) if t==1 and p==1])
    fp=sum([1 for t,p in zip(targets,predictions) if t==0 and p==1])
    fn=sum([1 for t,p in zip(targets,predictions) if t==1 and p==0])
    if tp + fp == 0 or tp + fn==0:
        f1_score = 0
    else:
        precision=tp/(tp+fp)
        recall=tp/(tp+fn)
        if precision ==0 or recall ==0:
            f1_score=0
        else:
            f1_score=2*(precision*recall)/(precision+recall)
    return f1_score
def compute_mae(targets,predictions):
    diff=targets-predictions
    abs_diff=torch.abs(diff)
    total_diff=torch.sum(abs_diff)
    mae=total_diff/targets.numel()
    return mae


def save2xt(f,float):
    formatted=[]
    for num in float:
        if num ==float[0]:
            formatt='{:.4e}'.format(num)
        else:
            formatt='{:.4f}'.format(num)
        formatted.append(formatt)
    line='{} {}'.format(formatted[0],formatted[1])
    f.write(line+'\n')
def evaluate_real(tt):
    num=0
    file = '{}.txt'.format(tt)
    with open(file, 'w') as f:
        f.write("MAE  F1  R2\n")
        total_loss=0
        total_f1=0
        list=[1,5,6,8,10]
        for i in list:
                  tcsv = fr"/home/wangmingyue/nangate_real/valid_case{i}/ir_drop_map.csv"
                  pcsv=fr"valid_case{i}output.csv"
                  if os.path.exists(tcsv):
                       num+=1
                       data1=np.loadtxt(tcsv,delimiter=",")
                       data2=np.loadtxt(pcsv,delimiter=",")
                       tensor1=torch.tensor(data1)# benchmark
                       tensor2=torch.tensor(data2)
                       #mae
                       mae=compute_mae(tensor1,tensor2)
                       maxdrop=torch.max(tensor1)
                       threshold=0.8*maxdrop
                       total_loss +=mae.item()
                       ###
                       tensor1=tensor1.view(-1)
                       tensor2=tensor2.view(-1)
                       pred=torch.where(tensor2 >=threshold,torch.tensor(1.),torch.tensor(0.))
                       labels = torch.where(tensor1 >= threshold, torch.tensor(1.), torch.tensor(0.))
                       f1=calculate_f1_score(pred,labels)
                       total_f1+=f1
                       reval = [mae.item(), f1]
                       save2xt(f, reval)
                       print(reval)
        print("total valid mae",total_loss/num)
        print("total valid f1 ", total_f1 / num)

def evaluate_fake(net,device,data_list,tdata_list,norm):
        tloss=0
        for x,y in zip(data_list,tdata_list):
            x=x.to(device=device,dtype=torch.float32)
            y=y.to(device=device,dtype=torch.float32)
            print("x",x.shape)
            pred=net(x)
            # print(norm)
            if norm is True:
                ytmean = [-mean / std for mean, std in zip(net.ymean, net.ystd)]
                ytstd = [1 / std for std in net.ystd]
                pred = transforms.Normalize(ytmean, ytstd)(pred)
            trainloss=compute_mae(y,pred)
            tloss+=trainloss.item()
        tloss=tloss/len(data_list)
        print(f'loss on training set: {tloss}')


def test_on_training(net,tdir):
    path = '/home/wangmingyue/Desktop/iccad/fake-circuit-data'
    x_data, y_data = read_fake_data(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm = False
    xmean, xstd = getxStat(x_data)
    trans_x = []
    for (x, y) in zip(x_data, y_data):
        trans_x.append(transforms.Normalize(xmean, xstd)(x))
    if tdir=='norm':
        norm = True
    evaluate_fake(net, device,trans_x, y_data,norm)
    return net

