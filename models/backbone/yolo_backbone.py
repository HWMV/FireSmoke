import torch
import torch.nn as nn
import math

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    @staticmethod
    def autopad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class YOLOv5Backbone(nn.Module):
    def __init__(self, input_channels=3, width_multiple=0.25, depth_multiple=0.33):
        super().__init__()
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple
        
        # Define channels and depths
        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [3, 6, 9, 3]
        
        # Apply multipliers
        channels = [self._make_divisible(c * width_multiple, 8) for c in base_channels]
        depths = [max(round(d * depth_multiple), 1) for d in base_depths]
        
        # Build backbone layers
        self.conv1 = Conv(input_channels, channels[0], 6, 2, 2)  # 0-P1/2
        self.conv2 = Conv(channels[0], channels[1], 3, 2)  # 1-P2/4
        self.c3_1 = C3(channels[1], channels[1], depths[0])
        self.conv3 = Conv(channels[1], channels[2], 3, 2)  # 3-P3/8
        self.c3_2 = C3(channels[2], channels[2], depths[1])
        self.conv4 = Conv(channels[2], channels[3], 3, 2)  # 5-P4/16
        self.c3_3 = C3(channels[3], channels[3], depths[2])
        self.conv5 = Conv(channels[3], channels[4], 3, 2)  # 7-P5/32
        self.c3_4 = C3(channels[4], channels[4], depths[3])
        self.sppf = SPPF(channels[4], channels[4], 5)  # 9
        
        self.out_channels = channels[2:]  # P3, P4, P5 channels

    @staticmethod
    def _make_divisible(x, divisor):
        return math.ceil(x / divisor) * divisor

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        p3 = self.c3_2(x)
        x = self.conv4(p3)
        p4 = self.c3_3(x)
        x = self.conv5(p4)
        x = self.c3_4(x)
        p5 = self.sppf(x)
        
        return [p3, p4, p5]  # Return P3/8, P4/16, P5/32