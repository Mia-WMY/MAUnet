from torchvision import transforms
import pandas as pd
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
class PixelwiseLoss(nn.Module):
    def forward(self,inputs,targets):
        return F.smooth_l1_loss(inputs,targets)
def l2_reg_loss(model,l2_alpha):
    l2_loss=[]
    for model in model.modules():
        if type(model) is nn.Conv2d:
            l2_loss.append((model.weight ** 2).sum()/2.0)
    return l2_alpha*sum(l2_loss)

def test_one_file(net,device,tdir,folder_path,pfile,i,norm):
    file_path = os.path.join(folder_path, pfile)
    # matrix_pth="/home/wangmingyue/real_case"
    # matrix="pitch{:d}.csv"
    # via_pth = "/home/wangmingyue/real_case"
    # via = "via{:d}.csv"
    # r_pth = "/home/wangmingyue/real_case"
    # r_patterns = "r{:d}.csv"
    logging.basicConfig(filename='test.log', level=logging.INFO,filemode='w')
    if os.path.exists(file_path):
        csv1 = os.path.join(folder_path, pfile, "current_map.csv")
        csv2 = os.path.join(folder_path, pfile, "eff_dist_map.csv")
        csv3 = os.path.join(folder_path, pfile, "pdn_density.csv")
        # mcsv= os.path.join(matrix_pth,matrix.format(i))
        # vcsv = os.path.join(via_pth, via.format(i))
        # rcsv=  os.path.join(r_pth, r_patterns.format(i))
        csv1 = pd.read_csv(csv1, header=None)
        csv2 = pd.read_csv(csv2, header=None)
        csv3 = pd.read_csv(csv3, header=None)
        # mcsv=pd.read_csv(mcsv,header=None)
        # vcsv = pd.read_csv(vcsv, header=None)
        # rcsv= pd.read_csv(rcsv, header=None)
        tsv1 = torch.tensor(csv1.values)
        esv1 = torch.unsqueeze(tsv1, dim=0)
        tsv2 = torch.tensor(csv2.values)
        esv2 = torch.unsqueeze(tsv2, dim=0)
        tsv3 = torch.tensor(csv3.values)
        esv3 = torch.unsqueeze(tsv3, dim=0)
        # mtsv=torch.tensor(mcsv.values)
        # mesv=torch.unsqueeze(mtsv,dim=0)
        # vtsv = torch.tensor(vcsv.values)
        # vesv = torch.unsqueeze(vtsv, dim=0)
        # rtsv = torch.tensor(rcsv.values)
        # resv = torch.unsqueeze(rtsv, dim=0)
        # concat = torch.cat((esv1, esv2, esv3,mesv,vesv,resv), dim=0)
        concat = torch.cat((esv1, esv2, esv3), dim=0)
        xdata=concat.to(device=device,dtype=torch.float32)
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
        np.savetxt(f"{pfile}output.csv",result,delimiter=",")


