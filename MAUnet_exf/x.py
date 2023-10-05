import pathlib
import sys
import os
import os.path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def init_ssf_scale_shift(dim):
    scale=nn.Parameter(torch.ones(dim))
    shift=nn.Parameter(torch.zeros(dim))
    nn.init.normal_(scale,mean=1,std=.02)
    nn.init.normal_(shift,std=.02)
    return scale,shift
def ssf_ada(x,scale,shift):
    assert scale.shape==shift.shape
    if x.shape[-1]==scale.shape[0]:
        return x*scale+shift
    elif x.shape[1]==scale.shape[0]:
        return x*scale.view(1,-1,1,1)+shift.view(1,-1,1,1)
    else:
        raise ValueError('the input tensor shape does not match scale')

class UNet_ssf(nn.Module):
    def __init__(self, n_channels, n_classes,xmax,xmin,ymax,ymin):
        super(UNet_ssf, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.xmax=xmax
        self.xmin=xmin
        self.ymax=ymax
        self.ymin=ymin
        self.bilinear = False
        self.tuning_mode='ssf'
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, self.bilinear))
        self.up2 = (Up(512, 256 // factor, self.bilinear))
        self.up3 = (Up(256, 128 // factor, self.bilinear))
        self.up4 = (Up(128, 64, self.bilinear))
        self.outc = (OutConv(64, n_classes))
        # if self.tuning_mode == 'ssf':
        self.ssf_scale1,self.ssf_shift1=init_ssf_scale_shift(dim=64)
        self.ssf_scale2, self.ssf_shift2 = init_ssf_scale_shift(dim=128)
        self.ssf_scale3, self.ssf_shift3 = init_ssf_scale_shift(dim=256)
        self.ssf_scale4, self.ssf_shift4 = init_ssf_scale_shift(dim=512)
        self.ssf_scale5, self.ssf_shift5 = init_ssf_scale_shift(dim=1024)
        self.ssf_scale6, self.ssf_shift6 = init_ssf_scale_shift(dim=512)
        self.ssf_scale7, self.ssf_shift7 = init_ssf_scale_shift(dim=256)
        self.ssf_scale8, self.ssf_shift8 = init_ssf_scale_shift(dim=128)
        self.ssf_scale9, self.ssf_shift9 = init_ssf_scale_shift(dim=64)


    def forward(self, x):
        # if self.tuning_mode == 'ssf':
            # print(x.shape)
            # print(self.inc(x).shape)
            # print(self.ssf_scale1.shape)
            # print(self.ssf_shift1.shape)
        x1 = ssf_ada(self.inc(x),self.ssf_scale1,self.ssf_shift1) # [1,64,x,x]
        x2 = ssf_ada(self.down1(x1),self.ssf_scale2,self.ssf_shift2)  # [1,128,x,x]
        x3 = ssf_ada(self.down2(x2),self.ssf_scale3,self.ssf_shift3) # [1,256,x,x]
        x4 = ssf_ada(self.down3(x3),self.ssf_scale4,self.ssf_shift4) # [1,512,x,x]
        x5 =ssf_ada( self.down4(x4),self.ssf_scale5,self.ssf_shift5)  # [1,1024,x,x]
        x =ssf_ada( self.up1(x5, x4),self.ssf_scale6,self.ssf_shift6)  # [1,512,x,x]
        x = ssf_ada(self.up2(x, x3) ,self.ssf_scale7,self.ssf_shift7) # [1,256,x,x]
        x =ssf_ada( self.up3(x, x2)  ,self.ssf_scale8,self.ssf_shift8)# [1,128,x,x]
        x =ssf_ada( self.up4(x, x1) ,self.ssf_scale9,self.ssf_shift9) # [1,64,x,x]
        # else:
        #     x1 = self.inc(x) # [1,64,x,x]
        #     x2 = self.down1(x1)# [1,128,x,x]
        #     x3 = self.down2(x2)# [1,256,x,x]
        #     x4 = self.down3(x3)# [1,512,x,x]
        #     x5 = self.down4(x4)# [1,1024,x,x]
        #     x = self.up1(x5, x4)# [1,512,x,x]
        #     x = self.up2(x, x3)# [1,256,x,x]
        #     x = self.up3(x, x2)#[1,128,x,x]
        #     x = self.up4(x, x1)#[1,64,x,x]
        logits = self.outc(x)
        return logits


def flip_matrix_up_down(matrix):
    return np.flipud(matrix)
def set_values_in_column(matrix, column_index, start_row, end_row, value):
    matrix[start_row:end_row + 1, column_index] = value
def find_connected_blocks(sorted_list, threshold):
    connected_blocks = []
    current_block = []
    for value in sorted_list:
        if not current_block or (value - current_block[-1]) <= threshold:
            current_block.append(value)
        else:
            connected_blocks.append(current_block)
            current_block = [value]
    # If there is any remaining block at the end of the loop, add it to the list.
    if current_block:
        connected_blocks.append(current_block)

    return connected_blocks
def read_csv_and_get_shape(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path,header=None)
        shape = df.shape
        return shape
    except FileNotFoundError:
        print("文件不存在或路径错误。")
        return None
    except pd.errors.EmptyDataError:
        print("CSV 文件为空。")
        return None
def get_ssf_pth():
    if getattr(sys,'frozen',False):
        base_path=sys._MEIPASS
    else:
        base_path=os.path.abspath(".")
    ssf_path=os.path.join(base_path,"ssf.pth")
    return ssf_path

def get_via_pitch_csv(x_pth):
    csv_pth = os.path.join(x_pth,'current_map.csv')
    netlist_pth = os.path.join(x_pth,'netlist.sp')
    csv_shape = read_csv_and_get_shape(csv_pth)
    grid4 = {}
    maxlength = csv_shape[0]  ##默认大小
    data_array_viar = [[0] * maxlength for _ in range(maxlength)]
    data_array_layer_r = [[0] * maxlength for _ in range(maxlength)]  # m1 m4 m7 m8 m9 合并的阻值
    with open(netlist_pth, 'r') as data:
        for temp_line in data:
            components = temp_line.split(" ")
            if components[0][0] == 'R':
                position1 = components[1].split("_")
                position2 = components[2].split("_")
                # 提取各个部分数据
                node1x = position1[2]  # 节点1
                node1y = position1[3]
                layer1 = position1[1][1]
                layer2 = position2[1][1]
                node2x= position2[2]
                node2y= position2[3]
                value = float(components[3])
                flag = 1
                if (layer1 != layer2):
                    x_round = round(int(node1x) / 2000) - 1
                    y_round = round(int(node1y) / 2000) - 1
                    data_array_viar[x_round][y_round] = value
                else:
                    if layer1 == '4':
                        node1xx = float(node1x) / 2000  # x坐标
                        node1yy = float(node1y) / 2000  # y坐标
                        node2yy = float(node2y) / 2000  # y坐标
                        if grid4.get(float(node1xx)) is None:
                            grid4[float(node1xx)] = []
                        grid4[float(node1xx)].append(float(node1yy))
                        grid4[float(node1xx)].append(float(node2yy))
                    if layer1 == '1':
                        y_round = math.floor(int(node1y) / 2000)
                        x_start = min(int(str(node1x)), int(str(node2x))) / 2000
                        x_start_round = math.floor(x_start)
                        x_end = max(int(str(node1x)), int(str(node2x))) / 2000  # 调整下数据的顺序
                        x_end_round = math.floor(x_end)
                        while (flag == 1):
                            seg_r1 = value / (x_end - x_start)
                            flag = 0  # 每层只执行第一次
                        i = x_end_round - x_start_round

                        data_array_layer_r[y_round][x_start_round] += seg_r1

                        while (i > 0):
                            data_array_layer_r[y_round][x_start_round + i] += seg_r1

                            i = i - 1
                    elif layer1 == '4':

                        x_round = math.floor(int(node1x) / 2000)
                        y_start = min(int(str(node1y)), int(str(node2y))) / 2000
                        y_start_round = math.floor(y_start)
                        y_end = max(int(str(node1y)), int(str(node2y))) / 2000  # 调整下数据的顺序
                        y_end_round = math.floor(y_end)
                        while (flag == 1):
                            seg_r4 = value / (y_end - y_start)
                            flag = 0  # 每层只执行第一次
                        i = y_end_round - y_start_round

                        data_array_layer_r[y_start_round][x_round] += seg_r4

                        while (i > 0):
                            data_array_layer_r[y_start_round + i][x_round] += seg_r4

                            i = i - 1

                    elif layer1 == '7':
                        y_round = math.floor(int(node1y) / 2000)
                        x_start = min(int(str(node1x)), int(str(node2x))) / 2000
                        x_start_round = math.floor(x_start)
                        x_end = max(int(str(node1x)), int(str(node2x))) / 2000  # 调整下数据的顺序
                        x_end_round = math.floor(x_end)
                        while (flag == 1):
                            seg_r7 = value / (x_end - x_start)
                            flag = 0  # 每层只执行第一次
                        i = x_end_round - x_start_round

                        data_array_layer_r[y_round][x_start_round] += seg_r7

                        while (i > 0):
                            data_array_layer_r[y_round][x_start_round + i] += seg_r7

                            i = i - 1

                    elif layer1 == '8':

                        x_round = math.floor(int(node1x) / 2000)
                        y_start = min(int(str(node1y)), int(str(node2y))) / 2000
                        y_start_round = math.floor(y_start)
                        y_end = max(int(str(node1y)), int(str(node2y))) / 2000  # 调整下数据的顺序
                        y_end_round = math.floor(y_end)
                        while (flag == 1):
                            seg_r8 = value / (y_end - y_start)
                            flag = 0  # 每层只执行第一次
                        i = y_end_round - y_start_round

                        data_array_layer_r[y_start_round][x_round] += seg_r8

                        while (i > 0):
                            data_array_layer_r[y_start_round + i][x_round] += seg_r8

                            i = i - 1

                    else:

                        y_round = math.floor(int(node1y) / 2000)
                        x_start = min(int(str(node1x)), int(str(node2x))) / 2000
                        x_start_round = math.floor(x_start)
                        x_end = max(int(str(node1x)), int(str(node2x))) / 2000  # 调整下数据的顺序
                        x_end_round = math.floor(x_end)
                        while (flag == 1):
                            seg_r9 = value / (x_end - x_start)
                            flag = 0  # 每层只执行第一次
                        i = x_end_round - x_start_round

                        data_array_layer_r[y_round][x_start_round] += seg_r9

                        while (i > 0):
                            data_array_layer_r[y_round][x_start_round + i] += seg_r9

                            i = i - 1
    matrix = np.zeros((maxlength, maxlength))
    for key, value in grid4.items():
        listx = list(set(value))
        listx.sort()
        threshold = 2.41
        connected_blocks_list = find_connected_blocks(listx, threshold)
        for idx, block in enumerate(connected_blocks_list, 1):
            set_values_in_column(matrix, int(key), int(block[0]), int(block[-1]), 1)
    pitch_csv=torch.tensor(matrix.copy())
    via_csv=torch.tensor(data_array_viar)
    r_csv=torch.tensor(data_array_layer_r)
    return pitch_csv,via_csv,r_csv


def run_one_file(net,file_pth):
    if os.path.exists(file_pth):
        csv1 = os.path.join(file_pth, "current_map.csv")
        csv2 = os.path.join(file_pth, "eff_dist_map.csv")
        csv3 = os.path.join(file_pth, "pdn_density.csv")
        mtsv,vtsv,rtsv=get_via_pitch_csv(file_pth)
        csv1 = pd.read_csv(csv1, header=None)
        csv2 = pd.read_csv(csv2, header=None)
        csv3 = pd.read_csv(csv3, header=None)
        tsv1 = torch.tensor(csv1.values)
        esv1 = torch.unsqueeze(tsv1, dim=0)
        tsv2 = torch.tensor(csv2.values)
        esv2 = torch.unsqueeze(tsv2, dim=0)
        tsv3 = torch.tensor(csv3.values)
        esv3 = torch.unsqueeze(tsv3, dim=0)
        mesv=torch.unsqueeze(mtsv,dim=0)
        vesv = torch.unsqueeze(vtsv, dim=0)
        resv = torch.unsqueeze(rtsv, dim=0)
        concat = torch.cat((esv1, esv2, esv3,mesv,vesv,resv), dim=0)
        xdata=concat.to(dtype=torch.float32)
        for d in range(6):
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
        pred =torch.squeeze(pred,dim=1)
        pred = torch.squeeze(pred, dim=0)
        result=pred.cpu().detach().numpy()
        np.savetxt(f"IR_drop_pred.csv",result,delimiter=",")
        print("predict successfully! results saved in IR_drop_pred.csv")

if __name__=='__main__':
    args=sys.argv[1:]
    tpath=None
    if '--path' in args:
        arg_index=args.index('--path')
        if arg_index+1 < len(args):
            tpath=args[arg_index+1]
    folder=pathlib.Path(__file__).parent.resolve()
    model_pth = f"{folder}/1_full.pth"
    checkpoint=torch.load(model_pth,map_location=torch.device('cpu'))
    model_state_dict=checkpoint['model_state_dict']
    xmax=checkpoint['xmax']
    xmin=checkpoint['xmin']
    ymax=checkpoint['ymax']
    ymin=checkpoint['ymin']
    model = UNet_ssf(n_channels=6,n_classes=1,xmax=xmax,xmin=xmin,ymax=ymax,ymin=ymin)
    model.load_state_dict(model_state_dict)
    if os.path.exists(tpath):
        run_one_file(model, tpath)


