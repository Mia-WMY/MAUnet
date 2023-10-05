import torch
from torch import nn
import torch.nn.functional as F
def ssf_ada(x,scale,shift):
    assert scale.shape==shift.shape
    if x.shape[-1]==scale.shape[0]:
        return x*scale+shift
    elif x.shape[1]==scale.shape[0]:
        return x*scale.view(1,-1,1,1)+shift.view(1,-1,1,1)
    else:
        raise ValueError('the input tensor shape does not match scale')
def init_ssf_scale_shift(dim):
    scale=nn.Parameter(torch.ones(dim))
    shift=nn.Parameter(torch.zeros(dim))
    nn.init.normal_(scale,mean=1,std=.02)
    nn.init.normal_(shift,std=.02)
    return scale,shift
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
    def __init__(self,in_channels, out_channels,xmax,xmin,ymax,ymin):
        super(UNet, self).__init__()
        #self.in_channels = in_channels
        # 卷积核
        filters = [64, 32, 16, 64]
        self.xmax=xmax
        self.xmin=xmin
        self.ymax = ymax
        self.ymin = ymin
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
        # if self.tuning_mode == 'ssf':
        self.ssf_scale1, self.ssf_shift1 = init_ssf_scale_shift(dim=filters[0])
        self.ssf_scale2, self.ssf_shift2 = init_ssf_scale_shift(dim=filters[1])
        self.ssf_scale3, self.ssf_shift3 = init_ssf_scale_shift(dim=filters[2])
        self.ssf_scale4, self.ssf_shift4 = init_ssf_scale_shift(dim=filters[3])
        self.ssf_scale5, self.ssf_shift5 = init_ssf_scale_shift(dim=filters[2])
        self.ssf_scale6, self.ssf_shift6 = init_ssf_scale_shift(dim=filters[1])
        self.ssf_scale7, self.ssf_shift7 = init_ssf_scale_shift(dim=filters[0])

    def forward(self, inputs):
        # if self.tuning_mode == 'ssf':
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        x1 = ssf_ada(maxpool1, self.ssf_scale1, self.ssf_shift1)  # [1,64,x,x]
        conv2 = self.conv2(x1)
        maxpool2 = self.maxpool2(conv2)
        x2 = ssf_ada(maxpool2, self.ssf_scale2, self.ssf_shift2)  # [1,32,x,x]
        conv3 = self.conv3(x2)
        maxpool3 = self.maxpool3(conv3)
        x3 = ssf_ada(maxpool3, self.ssf_scale3, self.ssf_shift3)  # [1,16,x,x]
        x4 = ssf_ada(self.bottleneck(x3), self.ssf_scale4, self.ssf_shift4)  # [1,64,x,x]
        x = ssf_ada(self.upconv3(conv3,x4), self.ssf_scale5, self.ssf_shift5)  # [1,16,x,x]
        x = ssf_ada(self.upconv2(conv2,x), self.ssf_scale6, self.ssf_shift6)  # [1,32,x,x]
        up = ssf_ada(self.upconv1(conv1,x), self.ssf_scale7, self.ssf_shift7)  # [1,64,x,x]
        dim = inputs.shape[-1]
        x = self.final(up)
        x = nn.Upsample(size=(dim, dim))(x)
        x=nn.Upsample(size=(dim,dim))(x)
        return x


class UNet_asap7(nn.Module):
    # in_channels:输入图片维度
    # out_channels:输出图片维度
    def __init__(self, in_channels, out_channels, xmax, xmin, ymax, ymin):
        super(UNet_asap7, self).__init__()
        # self.in_channels = in_channels
        # 卷积核
        filters = [64, 32, 16, 64]
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.conv1 = Unetconv(in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = Unetconv(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = Unetconv(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # bottleneck
        self.bottleneck = Unetconv(filters[2], filters[3])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.loss = PixelwiseLoss()

        self.upconv3 = Upconv(filters[3], filters[2])
        self.upconv2 = Upconv(filters[2], filters[1])
        self.upconv1 = Upconv(filters[1], filters[0])
        # # 插值
        # self.upconv3 = Unetconv(filters[3] + filters[2], filters[2])
        # self.upconv2 = Unetconv(filters[2] + filters[1], filters[1])
        # self.upconv1 = Unetconv(filters[1] + filters[0], filters[0])
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, kernel_size=1),
            # nn.Conv2d(1, 1, kernel_size=2,stride=1,padding=0),
        )
        # if self.tuning_mode == 'ssf':
        self.ssf_scale1, self.ssf_shift1 = init_ssf_scale_shift(dim=filters[0])
        self.ssf_scale2, self.ssf_shift2 = init_ssf_scale_shift(dim=filters[1])
        self.ssf_scale3, self.ssf_shift3 = init_ssf_scale_shift(dim=filters[2])
        self.ssf_scale4, self.ssf_shift4 = init_ssf_scale_shift(dim=filters[3])
        self.ssf_scale5, self.ssf_shift5 = init_ssf_scale_shift(dim=filters[2])
        self.ssf_scale6, self.ssf_shift6 = init_ssf_scale_shift(dim=filters[1])
        self.ssf_scale7, self.ssf_shift7 = init_ssf_scale_shift(dim=filters[0])

    def forward(self, inputs):
        # if self.tuning_mode == 'ssf':
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        x1 = ssf_ada(maxpool1, self.ssf_scale1, self.ssf_shift1)  # [1,64,x,x]
        conv2 = self.conv2(x1)
        maxpool2 = self.maxpool2(conv2)
        x2 = ssf_ada(maxpool2, self.ssf_scale2, self.ssf_shift2)  # [1,32,x,x]
        conv3 = self.conv3(x2)
        maxpool3 = self.maxpool3(conv3)
        x3 = ssf_ada(maxpool3, self.ssf_scale3, self.ssf_shift3)  # [1,16,x,x]
        x4 = ssf_ada(self.bottleneck(x3), self.ssf_scale4, self.ssf_shift4)  # [1,64,x,x]
        x = ssf_ada(self.upconv3(conv3, x4), self.ssf_scale5, self.ssf_shift5)  # [1,16,x,x]
        x = ssf_ada(self.upconv2(conv2, x), self.ssf_scale6, self.ssf_shift6)  # [1,32,x,x]
        up = ssf_ada(self.upconv1(conv1, x), self.ssf_scale7, self.ssf_shift7)  # [1,64,x,x]

        x = self.final(up)

        return x




