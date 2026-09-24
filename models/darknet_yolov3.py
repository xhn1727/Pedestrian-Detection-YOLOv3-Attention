import torch
import torch.nn as nn
import numpy as np

# 官方 Darknet 的 Conv-BN-Leaky
class ConvBNLeaky(nn.Module):
    def __init__(self, in_c, out_c, k, s=1):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 残差 Residual Block
class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        half = c // 2
        self.layer = nn.Sequential(
            ConvBNLeaky(c, half, 1),
            ConvBNLeaky(half, c, 3)
        )

    def forward(self, x):
        return x + self.layer(x)

# Darknet-53 主干
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.block0 = nn.Sequential(
            ConvBNLeaky(3, 32, 3),
            ConvBNLeaky(32, 64, 3, 2),
            Residual(64)
        )

        self.block1 = self._make_layer(64, 128, num=2)
        self.block2 = self._make_layer(128, 256, num=8)
        self.block3 = self._make_layer(256, 512, num=8)
        self.block4 = self._make_layer(512, 1024, num=4)

    def _make_layer(self, in_c, out_c, num):
        layers = [ConvBNLeaky(in_c, out_c, 3, 2)]
        for _ in range(num):
            layers.append(Residual(out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        out_52 = self.block2(x)
        out_26 = self.block3(out_52)
        out_13 = self.block4(out_26)
        return out_52, out_26, out_13

    # 加载 darknet53.conv.74 预训练权重
    def load_darknet_weights(self, file):
        print(f"Loading Darknet53 pretrained weights: {file}")

        with open(file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for m in self.modules():
            if isinstance(m, ConvBNLeaky):

                # BN bias, weight, running mean, var
                bn = m.bn
                num = bn.bias.numel()

                for param in [bn.bias, bn.weight, bn.running_mean, bn.running_var]:
                    size = param.numel()
                    param.data.copy_(torch.from_numpy(weights[ptr:ptr+size]).view_as(param))
                    ptr += size

                # Conv weights
                conv_w = m.conv.weight
                size = conv_w.numel()
                conv_w.data.copy_(torch.from_numpy(weights[ptr:ptr+size]).view_as(conv_w))
                ptr += size

        print("Darknet53 pretrained backbone loaded!")

# YOLOv3 Head 和主模型
class YOLOHead(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNLeaky(in_c, out_c, 1),
            ConvBNLeaky(out_c, in_c, 3),
            ConvBNLeaky(in_c, out_c, 1),
            ConvBNLeaky(out_c, in_c, 3),
            ConvBNLeaky(in_c, out_c, 1),
        )

    def forward(self, x):
        return self.layers(x)


class YOLOv3(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=None):
        super().__init__()

        self.backbone = Darknet53()
        if pretrained_backbone:
            self.backbone.load_darknet_weights(pretrained_backbone)

        # 三个检测头
        self.head1 = YOLOHead(1024, 512)
        self.out1 = nn.Conv2d(512, 3*(5+num_classes), 1)

        self.reduce1 = ConvBNLeaky(512, 256, 1)
        self.up1 = nn.Upsample(scale_factor=2)

        self.head2 = YOLOHead(256+512, 256)
        self.out2 = nn.Conv2d(256, 3*(5+num_classes), 1)

        self.reduce2 = ConvBNLeaky(256, 128, 1)
        self.up2 = nn.Upsample(scale_factor=2)

        self.head3 = YOLOHead(128+256, 128)
        self.out3 = nn.Conv2d(128, 3*(5+num_classes), 1)

    def forward(self, x):
        f52, f26, f13 = self.backbone(x)

        y1 = self.head1(f13)
        p1 = self.out1(y1)

        x = self.reduce1(y1)
        x = self.up1(x)
        x = torch.cat([x, f26], dim=1)
        y2 = self.head2(x)
        p2 = self.out2(y2)

        x = self.reduce2(y2)
        x = self.up2(x)
        x = torch.cat([x, f52], dim=1)
        y3 = self.head3(x)
        p3 = self.out3(y3)

        return p1, p2, p3