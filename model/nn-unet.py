import torch
import torch.nn as nn

class InConv(nn.Module):

    def __init__(self, in_chn, out_chn):
        self.conv1 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_chn, 2*out_chn, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_chn)
        self.norm2 = nn.InstanceNorm3d(out_chn)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inp):
        out = self.lrelu(self.norm1(self.conv1(inp[-1])))
        out = self.lrelu(self.norm2(self.conv2(out)))
        inp.append(out)
        return inp

class DownsampleBlock(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chn, in_chn, 2, 2)
        self.conv2 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.norm1 = nn.InstanceNorm3d(in_chn)
        self.norm2 = nn.InstanceNorm3d(out_chn)
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
        self.norm1 = nn.InstanceNorm3d(chn)
        self.norm2 = nn.InstanceNorm3d(chn)
        self.lrelu = nn.LeakyReLU()

    def forward(self, inp):
        out = self.lrelu(self.norm1(self.conv1(inp[-1])))
        out = self.lrelu(self.norm2(self.conv2(out)))
        inp.append(out)
        return inp

class UpsampleBlock(nn.Module):

    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_chn, out_chn, 2, 2)
        self.conv1 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_chn, out_chn, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(out_chn)
        self.lrelu = nn.LeakyReLU()

    def forward(self, skip_features, inp):
        next_skip = skip_features.pop(-1)
        out = self.lrelu(self.norm(self.deconv(inp)))
        out = torch.concat([next_skip, self.deconv(inp)])
        out = self.lrelu(self.norm(self.conv1(out)))
        out = self.lrelu(self.norm(self.conv2(out)))
        return skip_features, out

class OutConv(nn.Module):

    def __init__(self, in_chn, out_chn, num_classes):
        self.deconv = nn.ConvTranspose3d(in_chn, out_chn, 2, 2)
        self.conv1 = nn.Conv3d(in_chn, out_chn, 3, 1, 1)
        self.conv2 = nn.Conv3d(out_chn, out_chn, 3, 1, 1)
        self.conv3 = nn.Conv3d(out_chn, num_classes, 1, 1)
        self.norm = nn.InstanceNorm3d(in_chn)
        self.lrelu = nn.LeakyReLU()

    def forward(self, skip_features, inp):
        next_skip = skip_features.pop(-1)
        out = self.lrelu(self.norm(self.deconv(inp)))
        out = torch.concat([next_skip, self.deconv(inp)])
        out = self.lrelu(self.norm(self.conv1(inp)))
        out = self.lrelu(self.norm(self.conv2(inp)))
        out = self.conv3(inp)
        return out


class nnUnet(nn.Module):

    def __init__(self, init_chn: int, next_chn: int, num_blocks, num_classes):
        super().__init__()
        self.init_chn = init_chn
        self.next_chn = next_chn
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.build()

    def build(self):
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.in_layer = InConv(
            in_chn=self.init_chn,
            out_chn=self.next_chn
        )
        nc = 2*self.next_chn
        for i in range(self.num_blocks):
            nc = nc*(2**i),
            self.encoder.add_module(f"down{i + 1}", DownsampleBlock(
                in_chn=nc,
                out_chn=2*nc
            ))

        self.encoder.add_module("bot_layer", Bottleneck(
            chn=2*nc
        ))

        for i in range(self.num_blocks):
            nc = nc/(2**i)
            self.decoder.add_module(f"up{i + 1}", UpsampleBlock(
                in_chn=nc,
                out_chn=nc/2
            ))

        self.decoder.add_module("out_layer", OutConv(
            in_chn = nc/2,
            out_chn = nc/4,
            num_classes=self.num_classes
        ))
        return

    def forward(self, x):
        features = self.encoder([x])
        bot_features = features.pop(-1)
        out = self.decoder(features, bot_features)
        return out


