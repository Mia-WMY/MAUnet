import torch
def min_max_normalize(tensor,channel):
    trans=tensor.clone()
    for d in range(channel):
        x=torch.squeeze(tensor[:,d,:,:])
        min_val=x.min()
        max_val=x.max()
        normalized_tensor=(x-min_val)/(max_val-min_val)
        trans[:,d,:,:]=torch.unsqueeze(normalized_tensor,0)
    return trans

def min_max_normalize_all(tensor,channel,min_val,max_val):
    trans=tensor.clone()
    for d in range(channel):
        x=torch.squeeze(tensor[:,d,:,:])
        normalized_tensor=(x-min_val[d])/(max_val[d]-min_val[d])
        trans[:,d,:,:]=torch.unsqueeze(normalized_tensor,0)
    return trans
def find_max_min(xdata,channel):
    xmin=[]
    xmax=[]
    for d in range(channel):
        i=0
        min_val=None
        max_val=None
        for tensor in xdata:
            x=torch.squeeze(tensor[:,d,:,:])
            if i==0:
                min_val=x.min()
                max_val=x.max()
                i=i+1
            else:
                min_val=min(min_val,x.min())
                max_val=max(max_val,x.max())
        xmin.append(min_val)
        xmax.append(max_val)
    return xmin,xmax
def max_min_trans(x,channel=4):
    trans=x.clone()
    for d in range(channel):
        tx=torch.squeeze(x[:,d,:,:])
        max=tx.max()
        min=tx.min()
        normal=(tx-min)/(max-min)
        trans[:,d,:,:]=torch.unsqueeze(normal,0)
    return trans
def getxStat(xdata,channel):
    min_val,max_val=find_max_min(xdata,channel=channel)
    trans_x=[]
    for x in xdata:
       transx=min_max_normalize_all(x,channel,min_val,max_val)
       trans_x.append(transx)
    # print(mean.numpy(),std.numpy())
    return trans_x,min_val,max_val

def get_norm(xdata,channel,min_val,max_val):
    trans_x=[]
    for x in xdata:
       transx=min_max_normalize_all(x,channel,min_val,max_val)
       trans_x.append(transx)
    return trans_x

def getyStat(xdata):
    min_val,max_val=find_max_min(xdata,1)
    trans_x=[]
    for x in xdata:
        transx=min_max_normalize_all(x,1,min_val,max_val)
        trans_x.append(transx)
    return trans_x,min_val,max_val
# from tools.data_process import read_data
# from torch.utils.data import random_split,DataLoader
# from train_net import train_net,train_net_new,test_net_new,test_one_file
# from tools.data_process import read_real_data,read_fake_data
# ####
# # str_x="*current_map.csv"
# # path='/home/wangmingyue/Desktop/iccad/asap7_data/'
# # str_y="*voltage_map_regular.csv"
# # path='/home/wangmingyue/Desktop/iccad/asap7_data/'
# path='/home/wangmingyue/Desktop/iccad/fake-circuit-data'
# tpath='/home/wangmingyue/Desktop/iccad/real-circuit-data/'
# x_data,y_data=read_fake_data(path)
# test_xdata,test_ydata=read_real_data(tpath)
# val_percent=0.9
# torch.cuda.empty_cache()
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# ####
# # str_x="*current_map.csv"
# # path='/home/wangmingyue/Desktop/iccad/asap7_data/'
# # str_y="*voltage_map_regular.csv"
# # path='/home/wangmingyue/Desktop/iccad/asap7_data/'
# path='/home/wangmingyue/Desktop/iccad/fake-circuit-data'
# tpath='/home/wangmingyue/Desktop/iccad/real-circuit-data/'
# x_data,y_data=read_fake_data(path)
# test_xdata,test_ydata=read_real_data(tpath)
# val_percent=0.9
# torch.cuda.empty_cache()
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # print(x_data[0].shape,y_data[0].shape)# list
# # xmean,xstd=getxStat(x_data)
# # ymean,ystd=getyStat(y_data)
# # xtrans=transforms.Compose([
# #     transforms.Normalize(xmean,xstd),
# # ])
# # ytrans=transforms.Compose([
# #     transforms.Normalize(ymean,ystd),
# # ])
# # trans_x=[]
# # trans_y=[]
# # for (cadc1063_alpha,y) in zip(x_data,y_data):
# #     trans_x.append(transforms.Normalize(xmean,xstd)(cadc1063_alpha))
# #     trans_y.append(transforms.Normalize(ymean,ystd)(y))
