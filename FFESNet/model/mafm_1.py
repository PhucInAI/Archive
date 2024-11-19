import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Attention Gate Module
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(gating_channels, in_channels, 1, stride=1, padding=0)
        self.W_x = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.psi = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(self.psi(g1 + x1))
        return x * psi

# Multi-Scale Attention Fusion Module (MAFM)
# Multi-Scale Attention Fusion Module with Pixel Shuffle
class MultiScaleAttentionFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels, upscaling_factor=2):
        super(MultiScaleAttentionFusionModule, self).__init__()

        self.num_align_channels = min(in_channels_list)

        # 1x1 convolutions to align feature map channels
        self.align_channels = nn.ModuleList([nn.Conv2d(in_channels, self.num_align_channels, kernel_size=1)
                                             for in_channels in in_channels_list])

        # Sub-Upscaling Modules (SUMs) for each scale to upscale to H/4 * W/4
        self.upscale_modules = nn.ModuleList([
            nn.ConvTranspose2d(self.num_align_channels, self.num_align_channels,
                               kernel_size=(upscaling_factor ** (i + 1)) * 2,
                               stride=upscaling_factor ** (i + 1),
                               padding=upscaling_factor ** (i + 1) // 2)
            for i in range(len(in_channels_list)-1)
        ])
        # for i in range(len(in_channels_list)-1):
        #     print((upscaling_factor ** (i + 1)) * 2)
        #     print(upscaling_factor ** (i + 1))
        #     print(upscaling_factor ** (i + 1) // 2)

        # # CBAM - Channel and Spatial Attention
        # self.channel_attention = ChannelAttention(in_channels_list[0])
        # self.spatial_attention = SpatialAttention()
        self.cbam = CBAM(self.num_align_channels)

        # 1x1 Convolution for dimensionality reduction after concatenation
        self.conv1x1 = nn.Conv2d(in_channels_list[0] * len(in_channels_list), in_channels_list[0], kernel_size=1)

        # Attention Gate
        self.attention_gate = AttentionGate(in_channels_list[0], in_channels_list[0])

        # Final Pixel Shuffle Layer for upsampling
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)  # Upscale by 4x

        # After pixel shuffle, adjust channels to match the output classes
        self.adjust_channels = nn.Conv2d(in_channels_list[0] // 16, out_channels, kernel_size=1)

    def forward(self, feature_maps):
        # Align channels of each feature map
        aligned_maps = [self.align_channels[i](feature_maps[i]) for i in range(len(feature_maps))]

        # for i in aligned_maps:
        #     print(i.shape)

        # for i in range(1, len(feature_maps)):
        #     tmp = self.upscale_modules[i-1](aligned_maps[i])
        #     print(tmp.shape)

        # Upscale feature maps 2, 3, 4 to the size of feature map 1 (H/4 * W/4)
        upscaled_maps = [aligned_maps[0]]
        for i in range(len(aligned_maps)-1):
            upscaled_maps += [self.upscale_modules[i](aligned_maps[i+1])]
        # upscaled_maps = [feature_maps[0]] + [self.upscale_modules[i-1](feature_maps[i])
        #                                      for i in range(len(feature_maps)-1)]

        # for i in upscaled_maps:
        #     print(i.shape)

        # Concatenate the upscaled feature maps along the channel dimension
        concatenated_features = torch.cat(upscaled_maps, dim=1)

        # Apply a 1x1 convolution to reduce dimensionality
        fused_features = self.conv1x1(concatenated_features)
        # print(fused_features.shape)

        # Apply Channel Attention and Spatial Attention (CBAM)
        # fused_features = self.channel_attention(fused_features)
        # fused_features = self.spatial_attention(fused_features)
        fused_features = self.cbam(fused_features)

        # Apply Attention Gate to the features
        # print(fused_features.shape)
        gated_features = self.attention_gate(fused_features, fused_features)

        # Upsample using Pixel Shuffle and adjust output channels
        shuffled_features = self.pixel_shuffle(gated_features)
        segmentation_map = self.adjust_channels(shuffled_features)

        return segmentation_map

# Example usage
input_feature_maps = [torch.randn(1, 64, 64, 64),   # H/4 * W/4
                      torch.randn(1, 128, 32, 32),  # H/8 * W/8
                      torch.randn(1, 256, 16, 16),  # H/16 * W/16
                      torch.randn(1, 512, 8, 8)]    # H/32 * W/32

# Initialize with different number of input channels for each feature map
mafm = MultiScaleAttentionFusionModule(in_channels_list=[64, 128, 256, 512], out_channels=1, upscaling_factor=2)
output_segmentation = mafm(input_feature_maps)

print(output_segmentation.shape)  # Expected output: (batch_size, num_classes, H, W)
