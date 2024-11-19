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

class MultiScaleAttentionFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleAttentionFusionModule, self).__init__()
        self.num_align_channels = min(in_channels_list)

        # 1x1 convolutions to align feature map channels
        self.align_channels = nn.ModuleList([nn.Conv2d(in_channels, self.num_align_channels, kernel_size=1)
                                             for in_channels in in_channels_list])

        # CBAM - Channel and Spatial Attention
        self.cbam = CBAM(self.num_align_channels)

        # 1x1 Convolution for dimensionality reduction after concatenation
        self.conv1x1 = nn.Conv2d(self.num_align_channels * len(in_channels_list), self.num_align_channels, kernel_size=1)

        # Final Pixel Shuffle Layer for upsampling
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=4)  # Upscale by 4x

        # After pixel shuffle, adjust channels to match the output classes
        self.adjust_channels = nn.Conv2d(self.num_align_channels // 16, out_channels, kernel_size=1)

    def forward(self, feature_maps):
        # Align channels of each feature map
        aligned_maps = [self.align_channels[i](feature_maps[i]) for i in range(len(feature_maps))]

        # Upscale feature maps to the size of feature map 1 (H/4 * W/4)
        upscaled_maps = [F.interpolate(aligned_maps[0], scale_factor=1, mode='bilinear', align_corners=False)]
        for i in range(1, len(aligned_maps)):
            scale_factor = 2 ** i
            upscaled_maps.append(F.interpolate(aligned_maps[i], scale_factor=scale_factor, mode='bilinear', align_corners=False))

        # Concatenate the upscaled feature maps along the channel dimension
        concatenated_features = torch.cat(upscaled_maps, dim=1)

        # Apply a 1x1 convolution to reduce dimensionality
        fused_features = self.conv1x1(concatenated_features)

        # Apply CBAM
        fused_features = self.cbam(fused_features)

        # Upsample using Pixel Shuffle and adjust output channels
        shuffled_features = self.pixel_shuffle(fused_features)
        segmentation_map = self.adjust_channels(shuffled_features)

        return segmentation_map

# Example usage
input_feature_maps = [torch.randn(1, 64, 64, 64),   # H/4 * W/4
                      torch.randn(1, 128, 32, 32),  # H/8 * W/8
                      torch.randn(1, 256, 16, 16),  # H/16 * W/16
                      torch.randn(1, 512, 8, 8)]    # H/32 * W/32

# Initialize with different number of input channels for each feature map
mafm = MultiScaleAttentionFusionModule(in_channels_list=[64, 128, 256, 512], out_channels=1)
output_segmentation = mafm(input_feature_maps)

print(output_segmentation.shape)  # Expected output: (batch_size, num_classes, H, W)
