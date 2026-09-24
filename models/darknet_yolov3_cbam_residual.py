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


class Residual_CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = conv_bn_leaky(c, c // 2, 1)
        self.conv2 = conv_bn_leaky(c // 2, c, 3)
        self.cbam  = CBAM(c)   # ← 在这里加最稳定

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = out + x
        out = self.cbam(out)   # ← CBAM 放 residual 后
        return out


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        self.block0 = nn.Sequential(
            ConvBNLeaky(3, 32, 3),
            ConvBNLeaky(32, 64, 3, 2),
            Residual_CBAM(64)
        )

        self.block1 = self._make_layer(64, 128, num=2)
        self.block2 = self._make_layer(128, 256, num=8)
        self.block3 = self._make_layer(256, 512, num=8)
        self.block4 = self._make_layer(512, 1024, num=4)

    def _make_layer(self, in_c, out_c, num):
        layers = [ConvBNLeaky(in_c, out_c, 3, 2)]
        for _ in range(num):
            layers.append(Residual_CBAM(out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        out_52 = self.block2(x)
        out_26 = self.block3(out_52)
        out_13 = self.block4(out_26)
        return out_52, out_26, out_13

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

        # ★ 新增：特征稳定层（解决 CBAM 导致的尺度变化）
        self.stabilizer = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 原 YOLO head 层（保持不变）
        self.layers = nn.Sequential(
            ConvBNLeaky(in_c, out_c, 1),
            ConvBNLeaky(out_c, in_c, 3),
            ConvBNLeaky(in_c, out_c, 1),
            ConvBNLeaky(out_c, in_c, 3),
            ConvBNLeaky(in_c, out_c, 1),
        )

    def forward(self, x):
        x = self.stabilizer(x)    # 让 head 适配 CBAM 过后的特征分布
        return self.layers(x)


# YOLOv3 主体结构
class YOLOv3_CBAM_RESIDUAL(nn.Module):
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