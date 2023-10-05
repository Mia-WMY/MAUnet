import argparse
import torch
from modules.normalization import  getxStat,getyStat
from modules.train import train_net
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from modules.multiscale import MSU_attNet
import random
def count(list1,list2):
    listy=[]
    for tensor1, tensor2 in zip(list1,list2):
        if tensor1.shape!=tensor2.shape:
            y = F.interpolate(tensor2, size=(tensor1.size(2), tensor1.size(3)), mode="bilinear")
            listy.append(y)
        else:
            listy.append(tensor2)
    return listy
def setup_seed(seed):
    seed=int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
def read_fake_data_sky130hd(folder_path):
    file_pattern=['BeGAN_{:03d}_current_map.csv','eff_dist_{:03d}.csv','pdn_density{:d}.csv']
    tfile_pattern='BeGAN_{:03d}_voltage_map_regular.csv'
    data_list=[]
    tdata_list=[]
    for i in range(780):
        tfile_path = os.path.join(folder_path, 'pdn_density{:d}.csv'.format(i))
        if(os.path.exists(tfile_path)):
            concat_data=None
            shape = 0
            for pattern in file_pattern:
                file_path=os.path.join(folder_path,pattern.format(i))
                csv_content=pd.read_csv(file_path,header=None)
                torch_content=torch.tensor(csv_content.values)
                if concat_data is None:
                    shape = torch_content.shape
                    expand_torch = torch.unsqueeze(torch_content, dim=0)
                    concat_data=expand_torch
                else:
                    ex_result=np.zeros(shape)
                    ex_result[:torch_content.shape[0],:torch_content.shape[1]]=torch_content
                    expand_torch = torch.unsqueeze(torch.tensor(ex_result), dim=0)
                    concat_data=torch.cat((concat_data,expand_torch),dim=0)
            concat_data=torch.unsqueeze(concat_data,0)
            data_list.append(concat_data)
            tfile_path = os.path.join(folder_path, tfile_pattern.format(i))
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch=torch.unsqueeze(texpand_torch,0)
            tdata_list.append(texpand_torch)
    return data_list,tdata_list


parser=argparse.ArgumentParser(description="a simple program for IR drop")
parser.add_argument("--normal",type=bool,required=True,help="normalize y")
parser.add_argument("--epoch",type=int,required=True,help="epoch")
parser.add_argument("--lr",type=float,required=True,help="learning rate")
parser.add_argument("--path",type=str,required=True,help="model path")
parser.add_argument("--seed",type=str,required=True,help="random seed")
args=parser.parse_args()
setup_seed(args.seed)
# the path of data
path='/home/wangmingyue/date/sky130hd'
# read data
x_data,y_data=read_fake_data_sky130hd(path)
y_data=count(x_data,y_data)
print(len(y_data))
# use cuda if it is available
print(torch.cuda.is_available())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# normalize
trans_x,xmin,xmax=getxStat(x_data,3)
_,ymin,ymax=getyStat(y_data)
net=MSU_attNet(img_ch=3,output_ch=1,xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin)
net.to(device=device)
total_params=sum(p.numel() for p in net.parameters())
print(f"tparams:{total_params}")
# training
model = train_net(net, device, trans_x, y_data, args.epoch, args.lr)
# save
model_pth='pths/{}.pth'.format(args.path)
torch.save({'model_state_dict': model.state_dict(), 'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin}, model_pth)