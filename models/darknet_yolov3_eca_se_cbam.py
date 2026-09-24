import torch
import torch.nn as nn
import torch.nn.functional as F

# ECA (Efficient Channel Attention)
class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)       # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1,-2))  # [B,1,C]
        y = self.sigmoid(y).transpose(-1,-2).unsqueeze(-1)
        return x * y


# SE (Squeeze and Excitation)
class SE(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x)
        return x * w

# CBAM (Channel + Spatial)
class ChannelAttention(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        max,_ = x.max(1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
# 基础卷积
class ConvBNLeaky(nn.Module):
    def __init__(self, in_c, out_c, k, s=1):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Darknet residual
class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvBNLeaky(c, c // 2, 1)
        self.conv2 = ConvBNLeaky(c // 2, c, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# Darknet53 + Multi-Scale Attention
class Darknet53_MSA(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBNLeaky(3, 32, 3)
        self.conv2 = ConvBNLeaky(32, 64, 3, 2)

        self.layer1 = self._make_layer(64, 1)
        self.conv3 = ConvBNLeaky(64, 128, 3, 2)
        self.layer2 = self._make_layer(128, 2)

        self.conv4 = ConvBNLeaky(128, 256, 3, 2)
        self.layer3 = self._make_layer(256, 8)
        self.att_52 = ECA()              # ✔ 52×52使用 ECA

        self.conv5 = ConvBNLeaky(256, 512, 3, 2)
        self.layer4 = self._make_layer(512, 8)
        self.att_26 = SE(512)            # ✔ 26×26使用 SE

        self.conv6 = ConvBNLeaky(512, 1024, 3, 2)
        self.layer5 = self._make_layer(1024, 4)
        self.att_13 = CBAM(1024)         # ✔ 13×13使用 CBAM

    def _make_layer(self, c, num):
        return nn.Sequential(*[Residual(c) for _ in range(num)])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.conv3(x)
        x = self.layer2(x)

        # 52x52
        x = self.conv4(x)
        x52 = self.layer3(x)
        x52 = self.att_52(x52)

        # 26x26
        x = self.conv5(x52)
        x26 = self.layer4(x)
        x26 = self.att_26(x26)

        # 13x13
        x = self.conv6(x26)
        x13 = self.layer5(x)
        x13 = self.att_13(x13)

        return x52, x26, x13
    
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


class YOLOv3_ECA_SE_CBAM(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=None):
        super().__init__()

        self.backbone = Darknet53_MSA()
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