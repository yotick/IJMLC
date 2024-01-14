import torch
from torch import nn
import torch.nn.functional as F

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class GFB(nn.Module):
    def __init__(self, ir_vis_channel,global_channel,out_channel):
        super(GFB, self).__init__()
        self.conv3 = convblock(ir_vis_channel, ir_vis_channel, 3, 1, 1)
        self.conv4 = convblock(ir_vis_channel, ir_vis_channel, 3, 1, 1)
        self.conv_globalinfo = convblock(global_channel, ir_vis_channel, 1, 1, 0)
        self.conv_out = convblock(ir_vis_channel * 3, out_channel, 1, 1, 0)
        self.ca = CA(ir_vis_channel)
        self.spatialAttention = SpatialAttention()
    def forward(self, ir, vis, global_info):
        cur_size = ir.size()[2:]
        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))
        ir = self.ca(ir) * ir
        ir = self.conv3(ir)
        vis = self.spatialAttention(vis) * vis
        vis = self.conv4(vis)
        fus = torch.cat((ir, vis, global_info), 1)
        return self.conv_out(fus)

class GFB2(nn.Module):
    def __init__(self, ir_vis_channel,global_channel,out_channel):
        super(GFB2, self).__init__()
        self.conv_globalinfo = convblock(global_channel, ir_vis_channel, 1, 1, 0)
        self.conv_out = convblock(ir_vis_channel * 3, out_channel, 1, 1, 0)
    def forward(self, ir, vis, global_info):
        cur_size = ir.size()[2:]
        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))
        fus = torch.cat((ir, vis, global_info), 1)
        return self.conv_out(fus)

class GlobalInfo(nn.Module):
    def __init__(self):
        super(GlobalInfo, self).__init__()
        self.cov1 = convblock(512, 256, 1, 1, 0)
        self.ca1 = CA(256)
        self.ca3 = CA(256)
        self.ca5 = CA(256)

        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(11),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(17),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.b5 = nn.Sequential(
            nn.AdaptiveMaxPool2d(23),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.cov3 = convblock(256, 128, 1, 1, 0)

    def forward(self, rgb, t):
        x_size=rgb.size()[2:]
        x = torch.cat((rgb, t), 1)
        x = self.cov1(x)
        b5 = self.b5(x)
        b3 = self.b3(b5)
        b1 = self.b1(b3)
        b3_size=b3.size()[2:]
        b5_size=b5.size()[2:]
        b1 = self.ca1(b1) * b1
        b3 = self.ca3(b3) * b3
        b5 = self.ca5(b5) * b5
        b1 = F.interpolate(b1, b3_size, mode='bilinear', align_corners=True)
        b3 = b3 + b1
        b3 = F.interpolate(b3, b5_size, mode='bilinear', align_corners=True)
        b5 = b5 + b3
        b5 = F.interpolate(b5, x_size, mode='bilinear', align_corners=True)
        x = self.cov3(b5)
        return x

class CA(nn.Module):
     def __init__(self, channel, reduction=16):
         super(CA, self).__init__()
         self.maxpool = nn.AdaptiveMaxPool2d(1)
         self.avgpool = nn.AdaptiveAvgPool2d(1)
         self.se = nn.Sequential(
             nn.Conv2d(channel, channel // reduction, 1, bias=False),
             nn.ReLU(),
             nn.Conv2d(channel // reduction, channel, 1, bias=False)
         )
         self.sigmoid = nn.Sigmoid()

     def forward(self, x):
         max_result = self.maxpool(x)
         avg_result = self.avgpool(x)
         max_out = self.se(max_result)
         avg_out = self.se(avg_result)
         output = self.sigmoid(max_out + avg_out)
         return output



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.global_info = GlobalInfo()

        self.gfb2_1 = GFB(128,128,64)
        self.gfb1_1 = GFB(64,64,64)
        self.conv = convblock(64, 1, 1, 1, 0)

    def forward(self, rgb, t):

        global_info = self.global_info(rgb[2], t[2])
        d5 = self.gfb2_1(t[1], rgb[1], global_info)
        d7 = self.gfb1_1(t[0], rgb[0], d5)
        d = self.conv(d7)
        return d


class Decoder_DIFM(nn.Module):
    def __init__(self):
        super(Decoder_DIFM, self).__init__()
        self.global_info = GlobalInfo()
        self.conv1 = convblock(384, 128, 1, 1, 0)
        self.conv2 = convblock(256, 128, 1, 1, 0)
        self.conv3 = convblock(128, 1, 1, 1, 0)

    def forward(self, rgb, t):

        global_info = self.global_info(rgb[2], t[2])
        global_info = F.interpolate(global_info, rgb[1].size()[2:], mode='bilinear', align_corners=True)
        d5 = self.conv1(torch.cat([t[1], rgb[1], global_info] , 1))
        d5 = F.interpolate(d5, rgb[0].size()[2:], mode='bilinear', align_corners=True)
        d7 = self.conv2(torch.cat([t[0], rgb[0], d5] , 1))
        d = self.conv3(d7)
        return d
class Decoder_SIFM(nn.Module):
    def __init__(self):
        super(Decoder_SIFM, self).__init__()
        self.gfb2_1 = GFB(128,128,64)
        self.gfb1_1 = GFB(64,64,64)
        self.conv1 = convblock(512, 128, 1, 1, 0)
        self.conv2 = convblock(64, 1, 1, 1, 0)
    def forward(self, rgb, t):

        global_info = torch.cat([rgb[2], t[2]], 1)
        global_info = self.conv1(global_info)
        d5 = self.gfb2_1(t[1], rgb[1], global_info)
        d7 = self.gfb1_1(t[0], rgb[0], d5)
        d = self.conv2(d7)
        return d
class vgg16_new(nn.Module):
    def __init__(self):
        super(vgg16_new, self).__init__()

        self.conv1_1 = convblock(1, 64, 1, 1, 0)
        self.conv1_2 = convblock(64, 64, 3, 1, 1)
        self.conv1_3 = convblock(64, 64, 3, 1, 1)
        self.conv1_4 = convblock(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = convblock(64, 128, 1, 1, 0)
        self.conv2_2 = convblock(128, 128, 3, 1, 1)
        self.conv2_3 = convblock(128, 128, 3, 1, 1)
        self.conv2_4 = convblock(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3_1 = convblock(128, 256, 1, 1, 0)
        self.conv3_2 = convblock(256, 256, 3, 1, 1)
        self.conv3_3 = convblock(256, 256, 3, 1, 1)
        self.conv3_4 = convblock(256, 256, 3, 1, 1)

    def forward(self, x):
        layer = []
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x1 = self.conv1_4(x)

        x = self.pool1(x1)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x2 = self.conv2_4(x)

        x = self.pool2(x2)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x3 = self.conv3_4(x)
        layer.append(x1)
        layer.append(x2)
        layer.append(x3)

        return layer


class DenseFuseNet(nn.Module):
    def __init__(self):
        super(DenseFuseNet, self).__init__()
        self.rgb_net = vgg16_new()
        self.t_net = vgg16_new()
        self.decoder = Decoder()

    def forward(self, rgb, t):

        rgb_f = self.rgb_net(rgb)
        t_f = self.t_net(t)
        d7 = self.decoder(rgb_f, t_f)
        return d7