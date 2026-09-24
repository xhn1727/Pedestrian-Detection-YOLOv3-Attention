import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM 模块
# Channel Attention 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


#  CBAM 总模块 
class CBAM(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

# Darknet Backbone（加入 CBAM）
def conv_bn_leaky(c1, c2, k=1, s=1):
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
        nn.BatchNorm2d(c2),
        nn.LeakyReLU(0.1, inplace=True)
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c = channels
        self.conv1 = conv_bn_leaky(c, c // 2, k=1)
        self.conv2 = conv_bn_leaky(c // 2, c, k=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Darknet53_CBAM(nn.Module):
    def __init__(self):
        super().__init__()

        # stem
        self.conv1 = conv_bn_leaky(3, 32, k=3)
        self.conv2 = conv_bn_leaky(32, 64, k=3, s=2)

        # residual layers
        self.layer1 = self._make_layer(64, 1)
        self.conv3 = conv_bn_leaky(64, 128, k=3, s=2)
        self.layer2 = self._make_layer(128, 2)

        self.conv4 = conv_bn_leaky(128, 256, k=3, s=2)
        self.layer3 = self._make_layer(256, 8)
        self.cbam3 = CBAM(256)   # 改进点：加入 CBAM（小尺度特征 52×52）

        self.conv5 = conv_bn_leaky(256, 512, k=3, s=2)
        self.layer4 = self._make_layer(512, 8)
        self.cbam4 = CBAM(512)   # 改进点：加入 CBAM（中尺度特征 26×26）

        self.conv6 = conv_bn_leaky(512, 1024, k=3, s=2)
        self.layer5 = self._make_layer(1024, 4)
        self.cbam5 = CBAM(1024)  # 改进点：加入 CBAM（大尺度特征 13×13）

    def _make_layer(self, channels, num_blocks):
        layers = nn.ModuleList()
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.conv3(x)
        x = self.layer2(x)

        # 52×52 小目标
        x = self.conv4(x)
        x3 = self.layer3(x)
        x3 = self.cbam3(x3)   # CBAM

        # 26×26 中目标
        x = self.conv5(x3)
        x4 = self.layer4(x)
        x4 = self.cbam4(x4)   # CBAM

        # 13×13 大目标
        x = self.conv6(x4)
        x5 = self.layer5(x)
        x5 = self.cbam5(x5)   # CBAM

        return x3, x4, x5

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

#检测头
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


# YOLOv3 主体结构
class YOLOv3_CBAM(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=None):
        super().__init__()

        self.backbone = Darknet53_CBAM()
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