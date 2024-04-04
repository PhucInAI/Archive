"""Simple Local Emphasis layer"""
import torch
from torch import nn
from torch.nn import functional as F


class LocalEnhanceModule(nn.Module):
    """Local Enhance Module"""
    def __init__(self, in_channels, out_channels, num_heads=4, kernel_size=3, padding=1, dilation=1, reduction_ratio=16):
        super(LocalEnhanceModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding, dilation)
        self.attention_heads = nn.ModuleList([nn.Conv2d(in_channels, out_channels//num_heads, kernel_size, padding, dilation) for _ in range(num_heads)])
        self.se = SEBlock(out_channels, reduction_ratio)  # SE block

    def forward(self, x):
        conv_out = self.conv(x)
        attention_outs = [F.sigmoid(head(x)) for head in self.attention_heads]
        # Concatenate attention maps
        attention_out = torch.cat(attention_outs, dim=1)
        attention_out = F.layer_norm(attention_out, attention_out.size()[1:])  # Layer normalization
        out = conv_out * attention_out
        out = self.se(out)  # SE block
        return out + x  # Residual connection


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Blocks
    https://github.com/Ksuryateja/LS-CNN
    """

    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


__all__ = [LocalEnhanceModule]
