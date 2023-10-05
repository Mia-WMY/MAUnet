import torch
from torch import nn
import torch.nn.functional as F


class PixelwiseLoss(nn.Module):
    def forward(self,inputs,targets):
        return F.smooth_l1_loss(inputs,targets)
def l2_reg_loss(model,l2_alpha):
    l2_loss=[]
    for model in model.modules():
        if type(model) is nn.Conv2d:
            l2_loss.append((model.weight ** 2).sum()/2.0)
    return l2_alpha*sum(l2_loss)

class Unetconv(nn.Module):  # unet卷积块，由两个卷积层构成
    def __init__(self, in_channels, out_channels):
        self.in_channels=in_channels
        self.out_channels=out_channels
        super(Unetconv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            #  eps，momentum,affine
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out


class Upconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upconv, self).__init__()
        self.conv = Unetconv(out_channels*2, out_channels)
        self.upconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, input_r, input_u):
        # input_u是下采样的特征图，先进行反卷积
        output_u = self.upconv1(input_u)
        # 计算反卷积结果在高度、宽度上的尺寸差异offset
        offset = output_u.size()[-1] - input_r.size()[-1]
        #在左右赏析四个方向上构造填充参数
        pad = [offset // 2, offset - offset // 2, offset // 2, offset - offset // 2]
        output_r = F.pad(input_r, pad)
        # skip connection
        #print(output_r.shape)
        #print(output_u.shape)
        return self.conv(torch.cat((output_u, output_r), dim=1))


class UNet(nn.Module):
    # in_channels:输入图片维度
    # out_channels:输出图片维度
    def __init__(self,in_channels, out_channels,xmean,xstd,ymean,ystd,norm):
        super(UNet, self).__init__()
        #self.in_channels = in_channels
        # 卷积核
        filters = [64, 32, 16, 64]
        self.xmean=xmean
        self.xstd=xstd
        self.ymean = ymean
        self.ystd = ystd
        self.norm=norm
        self.conv1=Unetconv(in_channels,filters[0])
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        self.conv2 = Unetconv(filters[0],filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = Unetconv(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # bottleneck
        self.bottleneck = Unetconv(filters[2], filters[3])
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.loss = PixelwiseLoss()
        # 上采样
        # self.decoder = nn.Sequential(
        #    Upconv(filters[3], filters[2]),
        #    Upconv(filters[2], filters[1]),
        #    Upconv(filters[1], filters[0])
        # )
        self.upconv3=Upconv(filters[3],filters[2])
        self.upconv2=Upconv(filters[2],filters[1])
        self.upconv1=Upconv(filters[1],filters[0])
        # # 插值
        # self.upconv3 = Unetconv(filters[3] + filters[2], filters[2])
        # self.upconv2 = Unetconv(filters[2] + filters[1], filters[1])
        # self.upconv1 = Unetconv(filters[1] + filters[0], filters[0])
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, kernel_size=1),
            # nn.Upsample(size=size),
        )

    def forward(self, inputs):
        # print(inputs.shape)# 821b 821 3
        dim=inputs.shape[-1]
        conv1= self.conv1(inputs)
        #print("conv1",conv1.shape)
        maxpool1 = self.maxpool1(conv1)
        #print(maxpool1.shape)
        conv2=self.conv2(maxpool1)
        #print(conv2.shape)
        maxpool2=self.maxpool2(conv2)
        #print(maxpool2.shape)
        conv3=self.conv3(maxpool2)
        #print(conv3.shape)
        maxpool3=self.maxpool3(conv3)
        #print(maxpool3.shape)
        center=self.bottleneck(maxpool3)

        #print("conv3",center)
        # up3 = self.upconv3(torch.cat((conv3,self.up(center)),dim=1))
        # up2 = self.upconv2(torch.cat((conv2, self.up(up3)), dim=1))
        # up1 = self.upconv1(torch.cat((conv1,self.up(up2)), dim=1))
        up3=self.upconv3(conv3,center)
        up2=self.upconv2(conv2,up3)
        up1=self.upconv1(conv1,up2)
        x = self.final(up1)
        #print(dim)
        x=nn.Upsample(size=(dim,dim))(x)
        #print(cadc1063_alpha.shape)
        return x



