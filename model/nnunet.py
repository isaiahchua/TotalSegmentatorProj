import torch
import torch.nn as nn

DEBUG = False

def PrintShape(func):
    def printing(*args, **kwargs):
        if DEBUG:
            out = func(*args, **kwargs)
            print(out.detach().shape)
        else:
            out = func(*args, **kwargs)
        return out
    return printing

class InConv(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chn, out_chn, 2, 2)
        self.conv2 = nn.Conv3d(out_chn, out_chn, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(out_chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    @PrintShape
    def forward(self, inp):
        out = self.lrelu(self.norm(self.conv1(inp)))
        out = self.lrelu(self.norm(self.conv2(out)))
        return out

class DownsampleBlock(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chn, in_chn, 2, 2)
        self.conv2 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_chn, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    @PrintShape
    def forward(self, inp):
        out = self.lrelu(self.norm1(self.conv1(inp)))
        out = self.lrelu(self.norm2(self.conv2(out)))
        return out

class Bottleneck(nn.Module):

    def __init__(self, chn):
        super().__init__()
        self.conv1 = nn.Conv3d(chn, chn, 2, 2)
        self.conv2 = nn.Conv3d(chn, chn, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    @PrintShape
    def forward(self, inp):
        out = self.lrelu(self.norm(self.conv1(inp)))
        out = self.lrelu(self.norm(self.conv2(out)))
        return out

class UpsampleBlock(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_chn, in_chn, 2, 2)
        self.conv1 = nn.Conv3d(2*in_chn, in_chn, 3, 1, 1)
        self.conv2 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_chn, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    @PrintShape
    def forward(self, inp, skip_features):
        out = self.lrelu(self.norm1(self.deconv(inp)))
        out = self.lrelu(self.norm1(self.conv1(torch.cat([out, skip_features], 1))))
        out = self.lrelu(self.norm2(self.conv2(out)))
        return out

class OutConv(nn.Module):

    def __init__(self, in_chn, num_classes):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_chn, in_chn, 2, 2)
        self.norm = nn.InstanceNorm3d(in_chn, affine=True)
        self.lrelu = nn.LeakyReLU()
        self.outconv = nn.Conv3d(in_chn, num_classes, 1, 1)

    @PrintShape
    def forward(self, inp):
        out = self.lrelu(self.norm(self.deconv(inp)))
        out = self.outconv(out)
        return out

class nnUnet(nn.Module):

    def __init__(self, channels, num_classes):
        super().__init__()
        self.chn = channels
        self.num_classes = num_classes
        self.build()

    def build(self):
        self.inlayer = InConv(self.chn[0], self.chn[1])
        self.down1 = DownsampleBlock(self.chn[1], self.chn[2])
        self.down2 = DownsampleBlock(self.chn[2], self.chn[3])
        self.down3 = DownsampleBlock(self.chn[3], self.chn[4])
        self.down4 = DownsampleBlock(self.chn[4], self.chn[5])
        self.botlayer = Bottleneck(self.chn[5])
        self.up1 = UpsampleBlock(self.chn[5], self.chn[4])
        self.up2 = UpsampleBlock(self.chn[4], self.chn[3])
        self.up3 = UpsampleBlock(self.chn[3], self.chn[2])
        self.up4 = UpsampleBlock(self.chn[2], self.chn[1])
        self.up5 = UpsampleBlock(self.chn[1], self.chn[0])
        self.outlayer = OutConv(self.chn[0], self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x):
        e1 = self.inlayer(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        out = self.botlayer(e5)
        out = self.up1(out, e5)
        out = self.up2(out, e4)
        out = self.up3(out, e3)
        out = self.up4(out, e2)
        out = self.up5(out, e1)
        out = self.outlayer(out)
        return out


