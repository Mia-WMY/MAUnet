
import argparse
import pandas as pd
import torch
import os
import numpy as np
from modules.multiscale import MSU_attNet
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
def count(list1,list2):
    listy=[]
    for tensor1, tensor2 in zip(list1,list2):
        if tensor1.shape!=tensor2.shape:
            y = F.interpolate(tensor2, size=(tensor1.size(2), tensor1.size(3)), mode="bilinear")
            listy.append(y)
        else:
            listy.append(tensor2)
    return listy

def save2xt(f,float):
    formatted=[]
    for num in float:
        if num ==float[0]:
            formatt='{:.4e}'.format(num)
        else:
            formatt='{:.4f}'.format(num)
        formatted.append(formatt)
    line='{} {} {} {}'.format(formatted[0],formatted[1],formatted[2],formatted[3])
    f.write(line+'\n')
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

def compute_rmse(targets,predictions):
    diff=targets-predictions
    abs_diff=abs(diff)
    mae = abs_diff / abs(targets)
    mae = torch.sum(mae) / mae.numel()
    return mae

def re_real(test_y):
    num=0
    total_loss=0
    total_f1=0
    total_r2=0
    total_com=0
    total_cc=0
    total_ssim=0
    file='result_asap7.txt'
    with open(file,'w') as f:
        f.write("RMSE  F1  R2\n")
        for i in range(800,1000):
              tcsv = "/home/wangmingyue/date/asap7/BeGAN_{:03d}_voltage_map_regular.csv".format(i)
              pcsv="{:02d}output.csv".format(i)
              if os.path.exists(tcsv):
                   num+=1
                   data2=np.loadtxt(pcsv,delimiter=",")
                   tensor2 = torch.tensor(data2)
                   tensor1=test_y[i-800]
                   #mae
                   mae=compute_rmse(tensor1,tensor2)
                   maxdrop=torch.max(tensor1)
                   threshold=0.8*maxdrop
                   total_loss +=mae.item()
                   ###
                   meanA = torch.mean(tensor1)
                   meanB = torch.mean(tensor2)
                   conv = torch.mean((tensor1 - meanA) * (tensor2 - meanB))
                   stda = torch.std(tensor1)
                   stdb = torch.std(tensor2)
                   cc = conv / (stda * stdb)
                   ###
                   r2 = r2_score(tensor2,tensor1)
                   tensor1 = torch.squeeze(tensor1, dim=0)
                   tensor1 = torch.squeeze(tensor1, dim=0)
                   tensor2 = torch.squeeze(tensor2, dim=0)
                   Anp = np.array(tensor1)
                   Bnp = np.array(tensor2)
                   maxA = np.max(Anp)
                   minA = np.min(Anp)
                   maxB = np.max(Bnp)
                   minB = np.min(Bnp)
                   matrix1 = (Anp - minA) / (maxA - minA)
                   matrix2 = (Bnp - minB) / (maxB - minB)
                   ssimx = ssim(matrix2, matrix1)
                   tensor1=tensor1.view(-1)
                   tensor2=tensor2.view(-1)
                   pred=torch.where(tensor2 >=threshold,torch.tensor(1.),torch.tensor(0.))
                   labels = torch.where(tensor1 >= threshold, torch.tensor(1.), torch.tensor(0.))
                   f1=calculate_f1_score(pred,labels)
                   reval=[mae.item(),f1,cc,ssimx]
                   total_f1+=f1
                   total_r2+=r2
                   total_ssim+=ssimx
                   total_cc+=cc
                   total_com+=mae.item()/maxdrop
                   save2xt(f,reval)
                   print(reval)
        av_ame=total_loss/num
        av_f1=total_f1/num
        av_cc=total_cc/num
        av_ssim=total_ssim/num
        average = [av_ame, av_f1, av_cc,av_ssim]
        save2xt(f, average)
        print(average)
def test_one(net,device,xdata,i,norm):
        xdata=xdata.to(device=device,dtype=torch.float32)
        if norm==True:
            for d in range(3):
                tx = torch.squeeze(xdata[d, :, :])
                max = net.xmax[d]
                min = net.xmin[d]
                normal = (tx - min) / (max - min)
                xdata[ d, :, :] = torch.unsqueeze(normal, 0)
            xdata=xdata.unsqueeze(dim=0)
            pred=net(xdata)
            for d in range(1):
                ty=torch.squeeze(pred[d,:,:])
                max=net.ymax[d]
                min=net.ymin[d]
                normal=ty*(max-min)+min
                pred[d,:,:]=normal
        else:
            xdata = xdata.unsqueeze(dim=0)
            pred = net(xdata)
        pred =torch.squeeze(pred,dim=1)
        pred = torch.squeeze(pred, dim=0)
        result=pred.cpu().detach().numpy()
        np.savetxt('{:02d}output.csv'.format(i),result,delimiter=",")
def read_fake_data_test_asap7(folder_path):
    file_pattern = ['BeGAN_{:03d}_current_map.csv', 'eff_dist_{:03d}.csv', 'pdn_density_{:d}.csv']
    tfile_pattern = 'BeGAN_{:03d}_voltage_map_regular.csv'
    data_list = []
    tdata_list = []
    for i in range(800,1001):
        tfile_path = os.path.join(folder_path, 'pdn_density_{:d}.csv'.format(i))
        if (os.path.exists(tfile_path)):
            concat_data = None
            shape = 0
            for pattern in file_pattern:
                file_path = os.path.join(folder_path, pattern.format(i))
                csv_content = pd.read_csv(file_path, header=None)
                torch_content = torch.tensor(csv_content.values)
                if concat_data is None:
                    shape = torch_content.shape
                    expand_torch = torch.unsqueeze(torch_content, dim=0)
                    concat_data = expand_torch
                else:
                    ex_result = np.zeros(shape)
                    ex_result[:torch_content.shape[0], :torch_content.shape[1]] = torch_content
                    expand_torch = torch.unsqueeze(torch.tensor(ex_result), dim=0)
                    concat_data = torch.cat((concat_data, expand_torch), dim=0)

            concat_data = torch.unsqueeze(concat_data, 0)
            data_list.append(concat_data)
            ####3
            tfile_path = os.path.join(folder_path, tfile_pattern.format(i))
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch = torch.unsqueeze(texpand_torch, 0)
            tdata_list.append(texpand_torch)

    return data_list, tdata_list


parser=argparse.ArgumentParser(description="a simple program for IR drop")
parser.add_argument("--tdir",type=str,required=True,help="normalize y")
parser.add_argument("--model",type=str,required=True,help="epoch")
args=parser.parse_args()
# choose device
device=torch.device('cpu')
print(device)
# load trained model
model_pth = f'pths/{args.model}.pth'
pth='/home/wangmingyue/date/asap7'
print(model_pth)
checkpoint=torch.load(model_pth)
model_state_dict=checkpoint['model_state_dict']
xmax=checkpoint['xmax']
xmin=checkpoint['xmin']
ymax=checkpoint['ymax']
ymin=checkpoint['ymin']
model=MSU_attNet(img_ch=3,output_ch=1,xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin)
model.load_state_dict(model_state_dict)
dirpth = f'results/{args.tdir}'
os.chdir(dirpth)
# read data
test_x,test_y= read_fake_data_test_asap7(pth)
test_y=count(test_x,test_y)
# testing
for i,x,y in zip(range(800,1000),test_x,test_y):
    x=x.squeeze()
    test_one(model, device, x, i,norm=True)
print("test over")
re_real(test_y)
