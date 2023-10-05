import argparse
from modules.process import read_fake_data_nangate45
import torch
from modules.normalization import  getxStat,getyStat
from modules.train import train_net
from modules.modules import UNet
import numpy as np
import random
def setup_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
parser=argparse.ArgumentParser(description="a simple program for IR drop")
parser.add_argument("--normal",type=bool,required=True,help="normalize y")
parser.add_argument("--epoch",type=int,required=True,help="epoch")
parser.add_argument("--lr",type=float,required=True,help="learning rate")
parser.add_argument("--path",type=str,required=True,help="model path")
parser.add_argument("--tune_mode",type=str,required=True,help="tune mode")
parser.add_argument("--seed",type=str,required=True,help="random seed")
args=parser.parse_args()
setup_seed(args.seed)


path='/home/wangmingyue/nangate45'
x_data,y_data=read_fake_data_nangate45(path)
print(torch.cuda.is_available())
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
trans_x,xmin,xmax=getxStat(x_data,3)
_,ymin,ymax=getyStat(y_data)
net=UNet(in_channels=3,out_channels=1,xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin)
net.to(device=device)
total_params=sum(p.numel() for p in net.parameters())
print(f"tparams:{total_params}")
model = train_net(net, device, trans_x, y_data, args.epoch, args.lr)
model_pth='pths/{}.pth'.format(args.path)
torch.save({'model_state_dict': model.state_dict(), 'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin}, model_pth)