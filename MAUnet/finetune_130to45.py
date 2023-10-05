import os.path
import numpy as np
from torch import optim
import torch
from modules.multiscale import MSU_attNet
from modules.process import read_real_dataT
from modules.normalization import  getxStat,getyStat,get_norm
import random
import argparse
import pandas as pd
from modules.finetune_evaluate import evaluate_real
from modules.finetune_test import valid_one_file
def compute_mae(targets,predictions):
    diff=targets-predictions
    abs_diff=torch.abs(diff)
    total_diff=torch.sum(abs_diff)
    mae=total_diff/targets.numel()
    return mae
def setup_seed(seed):
    seed=int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def compute_mse(targets,predictions):
    diff=(targets-predictions)**2
    total_diff=torch.sum(diff)
    mse=total_diff/targets.numel()
    return mse
def calculate_f1(targets,predictions):
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
def calculate_f1_score(targets,predictions):
    maxdrop = torch.max(targets)
    threshold = 0.9 * maxdrop
    tensor1 = targets
    tensor2 = predictions
    pred = torch.where(tensor2 >= threshold, torch.tensor(1.), torch.tensor(0.))
    labels = torch.where(tensor1 >= threshold, torch.tensor(1.), torch.tensor(0.))
    f1 = calculate_f1(pred, labels)
    return f1


def compute_sae(targets,predictions):
    diff=targets-predictions
    square=diff**2
    total_diff=torch.sum(square)
    sae=total_diff/targets.numel()
    return sae

def fine_tuning_cor(net,lr,epoch,num):
    npath = '/home/wangmingyue/nangate_real'
    test_xdata, test_ydata = read_real_dataT(npath,num)
    checkpoint = torch.load(net)
    model_state_dict = checkpoint['model_state_dict']
    _, xmin, xmax = getxStat(test_xdata, 3)
    _, ymin, ymax = getyStat(test_ydata)
    xmin[2]=0
    xmax[2]=3
    xmax = checkpoint['xmax']
    xmin = checkpoint['xmin']
    ymax = checkpoint['ymax']
    ymin = checkpoint['ymin']
    finetune_xdata = get_norm(test_xdata, 3, xmin, xmax)
    pretrained_model = MSU_attNet(img_ch=3, output_ch=1, xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin)
    pretrained_model.load_state_dict(model_state_dict)
    pretrained_model.xmax=xmax
    pretrained_model.xmin=xmin
    pretrained_model.ymax=ymax
    pretrained_model.ymin=ymin
    for param in pretrained_model.parameters():
        param.requires_grad = False
    param_to_train=list(pretrained_model.Up4.parameters())+list(pretrained_model.outc.parameters())+list(pretrained_model.Up3.parameters())+list(pretrained_model.Up2.parameters())\
                      +list(pretrained_model.Upconv2.parameters())+list(pretrained_model.Upconv3.parameters())
                    # list(pretrained_model.Up4.parameters()) +
                   # # +list(pretrained_model.att4.parameters())\
                   #  + list(pretrained_model.att3.parameters())+list(pretrained_model.att2.parameters())
    for param in param_to_train:
        param.requires_grad=True
    unfrozen_name = ['ssf_scale1', 'ssf_shift1', 'ssf_scale2', 'ssf_shift2', 'ssf_scale3', 'ssf_shift3', 'ssf_scale4',
                     'ssf_shift4',]
    for name, param in pretrained_model.named_parameters():
        if name in unfrozen_name:
            param.requires_grad = True
    non_frozen=[param for param in pretrained_model.parameters() if param.requires_grad]
    optimizer = optim.Adam(non_frozen, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=50)
    pretrained_model.to(device)
    for e in range(0, epoch):
        tloss = 0
        pretrained_model.train()
        for x, y in zip(finetune_xdata, test_ydata):
            optimizer.zero_grad()
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            pred = pretrained_model(x)
            for d in range(1):
                ty = torch.squeeze(pred[d, :, :])
                max = ymax[d]
                min = ymin[d]
                normal = ty * (max - min) + min
                pred[d, :, :] = normal
            mae = compute_mae(pred, y)
            mse = mae
            trainloss = mse
            trainloss.backward()
            optimizer.step()
            tloss += trainloss.item()
        scheduler.step()
        print(f"epoch:{e},loss:{tloss}")
    return pretrained_model


parser=argparse.ArgumentParser(description="a simple program for IR drop")
parser.add_argument("--num",type=int,required=True,help="pretrained model")
parser.add_argument("--seed",type=str,required=True,help="random seed")
args=parser.parse_args()
setup_seed(args.seed)
# choose device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# finetune model
model_pth='pths/sky130hd_seed11.pth'
model=fine_tuning_cor(model_pth,1e-3,200,args.num)
# save model
model_pth='tfpths/sky130hd_{}_{}.pth'.format(args.num,args.seed)
torch.save({'model_state_dict': model.state_dict(), 'xmax': model.xmax, 'xmin': model.xmin, 'ymax': model.ymax, 'ymin': model.ymin}, model_pth)
tdir="tfpths/sky130hd_{}_{}".format(args.num,args.seed)
# load finetuned model
checkpoint=torch.load(model_pth)
model_state_dict=checkpoint['model_state_dict']
xmax=checkpoint['xmax']
xmin=checkpoint['xmin']
ymax=checkpoint['ymax']
ymin=checkpoint['ymin']
model = MSU_attNet(img_ch=3,output_ch=1,xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin)
model.load_state_dict(model_state_dict)
os.chdir(tdir)
device=torch.device('cpu')
# test data
vpath = '/home/wangmingyue/nangate_real'
valid_list=[1,5,6,8,10]
# testing loop
for i in valid_list:
    pfile = f"valid_case{i}"
    tfile = os.path.join(vpath, f"valid_case{i}")
    if os.path.exists(tfile):
        tcsv = os.path.join(vpath, pfile, "current_map.csv")
        csv1 = pd.read_csv(tcsv, header=None)
        valid_one_file(model, device, tdir,vpath, pfile, i,norm=True)
tt="sky130hd"
evaluate_real(tt)
