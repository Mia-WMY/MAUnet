import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=2, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                init.xavier_normal_(layer.weight)
    def forward(self, x):
        x = self.conv(x)
        return x


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
        for layer in self.double_conv:
            if isinstance(layer,nn.Conv2d):
                init.xavier_normal_(layer.weight)
    def forward(self, x):
        return self.double_conv(x)



class Upx(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
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
        for layer in self.conv1:
            if isinstance(layer, nn.Conv2d):
                init.xavier_normal_(layer.weight)
        for layer in self.conv2:
            if isinstance(layer, nn.Conv2d):
                init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Unetconv(out_channels * 2, out_channels)
        self.upconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        # print(">",x1.shape)# 1,64,102,102
        x1 = self.upconv1(x1)

        # print(x2.shape) # 1 16 205 205
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                init.xavier_normal_(layer.weight)

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_9(nn.Module):
    def __init_(self, ch_in, ch_out):
        super(conv_block_9, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=9, stride=1, padding=4, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_3_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_1, self).__init__()

        #self.conv_1 = conv_block_1(ch_in, ch_out)
        #self.conv_2 = conv_block_2(ch_in, ch_out)
        self.conv_3 = conv_block_3(ch_in, ch_out)
        #self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)
        #self.conv_9 = conv_block_9(ch_in, ch_out)

        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        #x1 = self.conv_1(x)
        #x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        #x5 = self.conv_5(x)
        x7 = self.conv_7(x)
        #x9 = self.conv_9(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x
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

class AttentionGate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionGate,self).__init__()
        self.W_g=nn.Sequential(
            nn.Conv2d(F_g,F_int,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        #  1024 2 2  0
        self.W_x=nn.Sequential(
            nn.Conv2d(F_l,F_int,kernel_size=2,stride=2,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi=nn.Sequential(
            nn.Conv2d(F_int,1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu=nn.ReLU(inplace=True)
    def forward(self,g,x):# [1 1024 51 51]   [1 512 102 102]
        g1=self.W_g(g)# 1 512 51 51
        x1=self.W_x(x)# 1 512 51 51
        if g1.shape !=x1.shape:
            diffY = g1.size()[2] - x1.size()[2]
            diffX = g1.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        psi=self.relu(g1+x1)# 1 512 51 51
        psi=self.psi(psi) # 1 1 51 51
        target=(x.shape[2],x.shape[3])
        up_psi=F.upsample(psi,size=target,mode='bilinear',align_corners=False)
        out=x*up_psi
        return out

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear = False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x1,x2):
        x1= self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])



        return x1



class MSU_Net(nn.Module):
    def __init__(self, img_ch, output_ch,xmax,xmin,ymax,ymin):
        super(MSU_Net, self).__init__()
        self.xmax=xmax
        self.xmin=xmin
        self.ymax=ymax
        self.ymin=ymin
        filters_number = [16,32,64,128]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_3_1(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_3_1(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[3])
        # self.Conv5 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[4])
        # self.Up5 = Up(filters_number[4], filters_number[3])
        self.Up4 = Up(filters_number[3], filters_number[2])
        self.Up3 = Up(filters_number[2], filters_number[1])
        self.Up2 = Up(filters_number[1], filters_number[0])
        self.outc = (OutConv(filters_number[0], output_ch))

        # self.ssf_scale1, self.ssf_shift1 = init_ssf_scale_shift(dim=filters_number[0])
        # self.ssf_scale2, self.ssf_shift2 = init_ssf_scale_shift(dim=filters_number[1])
        # self.ssf_scale3, self.ssf_shift3 = init_ssf_scale_shift(dim=filters_number[2])
        # self.ssf_scale4, self.ssf_shift4 = init_ssf_scale_shift(dim=filters_number[3])
        # self.ssf_scale5, self.ssf_shift5 = init_ssf_scale_shift(dim=filters_number[4])
        # self.ssf_scale6, self.ssf_shift6 = init_ssf_scale_shift(dim=filters_number[3])
        # self.ssf_scale7, self.ssf_shift7 = init_ssf_scale_shift(dim=filters_number[2])
        # self.ssf_scale8, self.ssf_shift8 = init_ssf_scale_shift(dim=filters_number[1])
        # self.ssf_scale9, self.ssf_shift9 = init_ssf_scale_shift(dim=filters_number[0])
        # #
        # self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        # self.Up_conv5 = conv_3_1(ch_in=filters_number[4], ch_out=filters_number[3])
        # 
        # self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        # self.Up_conv4 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[2])
        # 
        # self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        # self.Up_conv3 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[1])
        # 
        # self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        # self.Up_conv2 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[0])
        # 
        # self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)
        # x1 = ssf_ada(x1, self.ssf_scale1, self.ssf_shift1)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # x2 = ssf_ada(x2, self.ssf_scale2, self.ssf_shift2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # x3 = ssf_ada(x3, self.ssf_scale3, self.ssf_shift3)
        x4 = self.Maxpool(x3)
        x4= self.Conv4(x4)
        x = self.Up4(x4, x3)  # [1,64,x,x]
        x = self.Up3(x, x2) # [1,64,x,x]
        x = self.Up2(x, x1)  # [1,64,x,x]
        x = self.outc(x) # [1,64,x,x]

        return x

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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

class MSU_reNet(nn.Module):
    def __init__(self, img_ch, output_ch,xmax,xmin,ymax,ymin):
        super(MSU_reNet, self).__init__()
        self.xmax=xmax
        self.xmin=xmin
        self.ymax=ymax
        self.ymin=ymin
        self.bilinear = False
        filters_number = [64,32,16,8]
        filters = [64,32, 16, 8]
        self.conv1 = Unetconv(img_ch, filters_number[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = Unetconv(filters_number[0], filters_number[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = Unetconv(filters_number[1], filters_number[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = Unetconv(filters_number[2], filters_number[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        # bottleneck
        self.bottleneck = Unetconv(filters[3], filters[3])
        # ???
        # self.decoder = nn.Sequential(
        #    Upconv(filters[3], filters[2]),
        #    Upconv(filters[2], filters[1]),
        #    Upconv(filters[1], filters[0])
        # )
        self.upconv3 = Upconv(filters[3], filters[2])
        self.upconv2 = Upconv(filters[2], filters[1])
        self.upconv1 = Upconv(filters[1], filters[0])
        # # ??
        # self.upconv3 = Unetconv(filters[3] + filters[2], filters[2])
        # self.upconv2 = Unetconv(filters[2] + filters[1], filters[1])
        # self.upconv1 = Unetconv(filters[1] + filters[0], filters[0])
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], output_ch, kernel_size=1),
            # nn.Upsample(size=size),
        )
        self.ssf_scale1, self.ssf_shift1 = init_ssf_scale_shift(dim=filters[0])
        self.ssf_scale2, self.ssf_shift2 = init_ssf_scale_shift(dim=filters[1])
        self.ssf_scale3, self.ssf_shift3 = init_ssf_scale_shift(dim=filters[2])
        self.ssf_scale4, self.ssf_shift4 = init_ssf_scale_shift(dim=filters[3])
        self.ssf_scale5, self.ssf_shift5 = init_ssf_scale_shift(dim=filters[2])
        self.ssf_scale6, self.ssf_shift6 = init_ssf_scale_shift(dim=filters[1])
        self.ssf_scale7, self.ssf_shift7 = init_ssf_scale_shift(dim=filters[0])
        self.att3 = AttentionGate(F_g=filters_number[3], F_l=filters_number[2], F_int=filters_number[2])
        self.att2 = AttentionGate(F_g=filters_number[2], F_l=filters_number[1], F_int=filters_number[1])
        self.att1 = AttentionGate(F_g=filters_number[1], F_l=filters_number[0], F_int=filters_number[0])
        # self.att1 = AttentionGate(F_g=32, F_l=64, F_int=64)

    def forward(self, x):

        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        x1 = ssf_ada(maxpool1, self.ssf_scale1, self.ssf_shift1)  # [1,64,x,x]
        conv2 = self.conv2(x1)
        maxpool2 = self.maxpool2(conv2)
        x2 = ssf_ada(maxpool2, self.ssf_scale2, self.ssf_shift2)  # [1,32,x,x]
        conv3 = self.conv3(x2)
        maxpool3 = self.maxpool3(conv3)
        x3 = ssf_ada(maxpool3, self.ssf_scale3, self.ssf_shift3)  # [1,16,x,x]
        conv4 = self.conv4(x3)
        maxpool4 = self.maxpool4(conv4)
        x4 = ssf_ada(maxpool4, self.ssf_scale4, self.ssf_shift4)  # [1,1
        x=self.bottleneck(x4)

        out_att3 = self.att3(g=x, x=maxpool3)
        print(out_att3.shape)
        print(x.shape)
        x = ssf_ada(self.upconv3(out_att3, x), self.ssf_scale5, self.ssf_shift5)  # [1,16,x,x]
        out_att2 = self.att2(g=x, x=maxpool2)
        x = ssf_ada(self.upconv2(out_att2, x), self.ssf_scale6, self.ssf_shift6)  # [1,32,x,x]
        out_att1 = self.att1(g=x, x=maxpool1)
        up = ssf_ada(self.upconv1(out_att1, x), self.ssf_scale7, self.ssf_shift7)  # [1,64,x,x]
        x = self.final(up)
        # print("x",x.shape)
        # x1 = self.inc(x)# 1 64 821 821
        # x2 = self.Maxpool(x1)
        # # print("x1",x1.shape)
        # # x2 = ssf_ada(x1, self.ssf_scale1, self.ssf_shift1)  # [1,64,x,x]
        # x2 = self.Conv2(x2) # 1 32 410 410
        # x3 = self.Maxpool(x2) # 1 16 205 205
        # # print("x2",x2.shape)
        # # x3 = ssf_ada(x2, self.ssf_scale2, self.ssf_shift2)  # [1,64,x,x]
        # x3 = self.Conv3(x3)Conv3
        # x3 = self.Maxpool(x3) # 1 64 102 102
        # print("x3",x3.shape)
        # x4 = ssf_ada(x3, self.ssf_scale3, self.ssf_shift3)  # [1,64,x,x]
        # x4= self.Conv4(x4)
        # x = self.Maxpool(x4)
        # print("x",x.shape)
        # out_att3 = self.att3(g=x, x=x3)  # 1,16
        # d4 = self.Up4(x4,x3) # 1 16 205 205 current16
        # d4 = torch.cat((x3, d4), dim=1) # 1 32 205 205  current16 + feature16
        # d4 = self.Up_conv4(d4) # 32 -- 16  current
        #
        # d3 = self.Up3(d4,x2) # 1 32
        # d3 = torch.cat((x2, d3), dim=1) # 1 64
        # d3 = self.Up_conv3(d3) # 64 --32
        #
        # d2 = self.Up2(d3,x1) # 32--64
        # d2 = torch.cat((x1, d2), dim=1) # 64
        # d2 = self.Up_conv2(d2) # 64
        #
        # d1 = self.Conv_1x1(d2)
        # x2 = ssf_ada(self.down1(x1), self.ssf_scale2, self.ssf_shift2)  # [1,32，x,x]
        # x3 = ssf_ada(self.down2(x2), self.ssf_scale3, self.ssf_shift3)  # [1,16,x,x]
        # x = ssf_ada(self.down3(x3), self.ssf_scale4, self.ssf_shift4)  # [1,64,x,x]
        # out_att3 = self.att3(g=x, x=x3)  # 1,16
        # x = ssf_ada(self.up1(x, out_att3), self.ssf_scale5, self.ssf_shift5)  # 32
        # # print("up3",x.shape)
        # out_att2 = self.att2(g=x3, x=x2)  # 32
        # x = self.up2(x3, out_att2)  # 64
        # # print("up2", x.shape)
        # out_att1 = self.att1(g=x, x=x1)  # 32
        # x = self.up3(x, out_att1)# 64
        # # print("up1", x.shape)
        # logits = self.outc(x)
        # print("out", x.shape)

        return x






class MSU_sNet(nn.Module):
    def __init__(self, img_ch, output_ch,xmax,xmin,ymax,ymin):
        super(MSU_sNet, self).__init__()
        self.xmax=xmax
        self.xmin=xmin
        self.ymax=ymax
        self.ymin=ymin
        filters_number = [16,32,64]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_3_1(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_3_1(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[2])
        # self.Conv4 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[3])
        # self.Conv5 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[4])
        # self.Up5 = Up(filters_number[4], filters_number[3])
        # self.Up4 = Up(filters_number[3], filters_number[2])
        self.Up3 = Up(filters_number[2], filters_number[1])
        self.Up2 = Up(filters_number[1], filters_number[0])
        self.outc = (OutConv(filters_number[0], output_ch))

        # self.ssf_scale1, self.ssf_shift1 = init_ssf_scale_shift(dim=filters_number[0])
        # self.ssf_scale2, self.ssf_shift2 = init_ssf_scale_shift(dim=filters_number[1])
        # self.ssf_scale3, self.ssf_shift3 = init_ssf_scale_shift(dim=filters_number[2])
        # self.ssf_scale4, self.ssf_shift4 = init_ssf_scale_shift(dim=filters_number[3])
        # self.ssf_scale5, self.ssf_shift5 = init_ssf_scale_shift(dim=filters_number[4])
        # self.ssf_scale6, self.ssf_shift6 = init_ssf_scale_shift(dim=filters_number[3])
        # self.ssf_scale7, self.ssf_shift7 = init_ssf_scale_shift(dim=filters_number[2])
        # self.ssf_scale8, self.ssf_shift8 = init_ssf_scale_shift(dim=filters_number[1])
        # self.ssf_scale9, self.ssf_shift9 = init_ssf_scale_shift(dim=filters_number[0])
        # #
        # self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        # self.Up_conv5 = conv_3_1(ch_in=filters_number[4], ch_out=filters_number[3])
        #
        # self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        # self.Up_conv4 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[2])
        #
        # self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        # self.Up_conv3 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[1])
        #
        # self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        # self.Up_conv2 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[0])
        #
        # self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)
        # x1 = ssf_ada(x1, self.ssf_scale1, self.ssf_shift1)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # x2 = ssf_ada(x2, self.ssf_scale2, self.ssf_shift2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # x3 = ssf_ada(x3, self.ssf_scale3, self.ssf_shift3)
        # x4 = self.Maxpool(x3)
        # x4= self.Conv4(x4)
        # x = self.Up4(x4, x3)  # [1,64,x,x]
        x = self.Up3(x, x2) # [1,64,x,x]
        x = self.Up2(x, x1)  # [1,64,x,x]
        x = self.outc(x) # [1,64,x,x]

        return x
class decoder(nn.Module):
    def __init__(self,in_c,out_c):
        super(decoder,self).__init__()
        self.cb=nn.Sequential(
            nn.Conv2d(in_c,out_c,3,padding=1,bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            # nn.ConvTranspose2d(mid_channels,out_c,4,stride=2,padding=1),
            # nn.BatchNorm2d(num_features=out_c),
            # nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x=self.cb(x)
        return x
class CenterBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super(CenterBlock,self).__init__()
        mid_channels=int(in_c*2)
        self.cb=nn.Sequential(
            nn.Conv2d(in_c,mid_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,mid_channels,3,padding=1,bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(mid_channels,out_c,4,stride=2,padding=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x=self.cb(x)
        return x

class Attention(nn.Module):
    def __init__(self,fg,f1,fi):
        super(Attention,self).__init__()
        self.wg=nn.Sequential(
            nn.Conv2d(f1,fi,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(fi),
        )
        self.wx=nn.Sequential(
            nn.Conv2d(fg,fi,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(fi),
        )
        self.psi=nn.Sequential(
            nn.Conv2d(fi,1,kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu=nn.ReLU(inplace=True)
    def forward(self,g,x):
        g1=self.wg(g)
        x1=self.wx(x)
        offset = x1.size()[-1] - g1.size()[-1]
        # 在左右赏析四个方向上构造填充参数
        pad = [offset // 2, offset - offset // 2, offset // 2, offset - offset // 2]
        g1= F.pad(g1, pad)
        psi=self.relu(g1+x1)
        psi=self.psi(psi)
        out=x*psi
        return out

class up_attconv(nn.Module):
    def __init__(self, ch_in, ch_out, ct=True):
        super(up_attconv, self).__init__()
        if ct:
            self.up = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=4, stride=2,padding=1)
        self.Conv=nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x,tx):
        x= self.up(x)
        x=self.Conv(x)
        offset = tx.size()[-1] - x.size()[-1]
        # 在左右赏析四个方向上构造填充参数
        pad = [offset // 2, offset - offset // 2, offset // 2, offset - offset // 2]
        x = F.pad(x, pad)
        return x


class MSU_attNet(nn.Module):
    def __init__(self, img_ch, output_ch,xmax,xmin,ymax,ymin):
        super(MSU_attNet, self).__init__()
        self.xmax=xmax
        self.xmin=xmin
        self.ymax=ymax
        self.ymin=ymin
        filters_number = [16,32,64,128]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_3_1(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_3_1(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[3])
        # self.center = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[3])
        # self.Conv5 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[4])
        self.Up4=up_attconv(ch_in=filters_number[3],ch_out=filters_number[2])
        self.Up3 = up_attconv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up2 = up_attconv(ch_in=filters_number[1], ch_out=filters_number[0])




        self.Upconv4 = decoder(in_c=filters_number[3],out_c=filters_number[2])
        self.Upconv3 = decoder(in_c=filters_number[2],out_c=filters_number[1])
        self.Upconv2 =decoder(in_c=filters_number[1],out_c=filters_number[0])
        self.outc = nn.Conv2d(filters_number[0],output_ch,kernel_size=1,stride=1,padding=0)

        self.att4 = Attention(fg=filters_number[2],f1=filters_number[2],fi=filters_number[1])
        self.att3 = Attention(fg=filters_number[1], f1=filters_number[1], fi=filters_number[0])
        self.att2 = Attention(fg=filters_number[0], f1=filters_number[0], fi=filters_number[0]//2)

    def forward(self, x):

        x1 = self.Conv1(x)
        # x1 = ssf_ada(x1, self.ssf_scale1, self.ssf_shift1)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # x2 = ssf_ada(x2, self.ssf_scale2, self.ssf_shift2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # x3 = ssf_ada(x3, self.ssf_scale3, self.ssf_shift3)
        x4 = self.Maxpool(x3)
        x4= self.Conv4(x4)
        ##
        # x5=self.Maxpool(x4)
        # x=self.center(x5)
        d4=self.Up4(x4,x3)
        x3=self.att4(g=d4,x=x3)
        d4=torch.cat((x3,d4),dim=1)
        d4=self.Upconv4(d4)
        ####
        d3 = self.Up3(d4,x2)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Upconv3(d3)
        ####
        d2 = self.Up2(d3,x1)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Upconv2(d2)
        ####

        x = self.outc(d2) # [1,64,x,x]

        return x
