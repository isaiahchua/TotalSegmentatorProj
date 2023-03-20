import torch
import torch.nn as nn

class InConv(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_chn, out_chn, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(out_chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inp):
        out = self.lrelu(self.norm(self.conv1(inp[-1])))
        out = self.lrelu(self.norm(self.conv2(out)))
        inp.append(out)
        return inp

class DownsampleBlock(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chn, in_chn, 2, 2)
        self.conv2 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_chn, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inp):
        out = self.lrelu(self.norm1(self.conv1(inp[-1])))
        out = self.lrelu(self.norm2(self.conv2(out)))
        inp.append(out)
        return inp

class Bottleneck(nn.Module):

    def __init__(self, chn):
        super().__init__()
        self.conv1 = nn.Conv3d(chn, chn, 2, 2)
        self.conv2 = nn.Conv3d(chn, chn, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inp):
        out = self.lrelu(self.norm(self.conv1(inp[-1])))
        out = self.lrelu(self.norm(self.conv2(out)))
        inp.append(out)
        return inp

class UpsampleBlock(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_chn, in_chn, 2, 2)
        self.conv1 = nn.Conv3d(2*in_chn, in_chn, 3, 1, 1)
        self.conv2 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_chn, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inp):
        out = self.lrelu(self.norm1(self.deconv(inp[-1])))
        out = torch.cat([inp[-2], out], 1)
        out = self.lrelu(self.norm1(self.conv1(out)))
        out = self.lrelu(self.norm2(self.conv2(out)))
        inp = inp[:-2]
        inp.append(out)
        return inp

class OutConv(nn.Module):

    def __init__(self, in_chn, out_chn, num_classes):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_chn, in_chn, 2, 2)
        self.conv1 = nn.Conv3d(2*in_chn, in_chn, 3, 1, 1)
        self.conv2 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.conv3 = nn.Conv3d(out_chn, num_classes, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_chn, affine=True)
        self.norm2 = nn.InstanceNorm3d(out_chn, affine=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inp):
        out = self.lrelu(self.norm1(self.deconv(inp[-1])))
        out = torch.cat([inp[-2], out], 1)
        out = self.lrelu(self.norm1(self.conv1(out)))
        out = self.lrelu(self.norm2(self.conv2(out)))
        out = self.conv3(out)
        return out


class nnUnet(nn.Module):

    def __init__(self, channels, num_classes):
        super().__init__()
        self.chn = channels
        self.num_classes = num_classes
        self.build()

    def build(self):
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.encoder.add_module("in_layer", InConv(
            in_chn=self.chn[0],
            out_chn=self.chn[1]
        ))

        for i in range(1, len(self.chn) - 1):
            self.encoder.add_module(f"down{i}", DownsampleBlock(
                in_chn=self.chn[i],
                out_chn=self.chn[i+1]
            ))

        self.encoder.add_module("bot_layer", Bottleneck(
            chn=self.chn[-1]
        ))

        for i in range(len(self.chn) - 1, 1, -1):
            self.decoder.add_module(f"up{len(self.chn) - i}", UpsampleBlock(
                in_chn=self.chn[i],
                out_chn=self.chn[i-1]
            ))

        self.decoder.add_module("out_layer", OutConv(
            in_chn = self.chn[1],
            out_chn = self.chn[0],
            num_classes=self.num_classes
        ))

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
        features = self.encoder([x])
        out = self.decoder(features)
        return out


