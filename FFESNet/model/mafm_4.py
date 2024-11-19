import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified Self Attention Block
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, width * height)
        attention = F.softmax(torch.bmm(proj_query, proj_key), dim=-1)
        proj_value = self.value_conv(x).view(batch, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch, C, width, height)
        return self.gamma * out + x  # Residual connection

# Simplified Attention Gate (Shared for all scales)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = F.interpolate(self.W_g(g), size=x.size()[2:], mode='bilinear', align_corners=False)
        psi = self.sigmoid(self.psi(F.relu(g1 + self.W_x(x))))
        return x * psi

# Multi-Scale Attention Fusion Module (With all scales)
class MultiScaleAttentionFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleAttentionFusionModule, self).__init__()
        self.num_align_channels = sum(in_channels_list)//8

        # Keep all dilated convolutions for channel alignment
        self.align_channels = nn.ModuleList([
            nn.Conv2d(in_channels, self.num_align_channels, kernel_size=3, padding=d, dilation=d)
            for in_channels, d in zip(in_channels_list, [1, 2, 4, 8])
        ])

        # Use a shared attention gate
        self.attention_gate = AttentionGate(self.num_align_channels, self.num_align_channels, self.num_align_channels // 2)

        # 1x1 Convolution for fusion with BatchNorm
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.num_align_channels * len(in_channels_list), self.num_align_channels, kernel_size=1),
            nn.BatchNorm2d(self.num_align_channels)
        )

        # Simplified Self-Attention
        self.self_attention = SelfAttention(self.num_align_channels)

        # Transposed Convolutions for upsampling (ModuleList for stacked transposed convolutions)
        self.upsampling_layers = nn.ModuleList([
            nn.ConvTranspose2d(self.num_align_channels, self.num_align_channels, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.num_align_channels, self.num_align_channels, kernel_size=4, stride=2, padding=1)
        ])

        # Final output layer with BatchNorm
        self.adjust_channels = nn.Sequential(
            nn.Conv2d(self.num_align_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, feature_maps):
        # Align feature map channels
        aligned_maps = [self.align_channels[i](feature_maps[i]) for i in range(len(feature_maps))]

        # Use a shared attention gate
        gated_maps = [self.attention_gate(aligned_maps[0], aligned_maps[i]) for i in range(len(aligned_maps))]

        # Upsample and align all maps to the largest scale (feature_maps[0] size)
        upscaled_maps = [F.interpolate(gated_maps[i], size=feature_maps[0].size()[2:], mode='bilinear', align_corners=False)
                         for i in range(len(gated_maps))]

        # Concatenate the feature maps
        concatenated_features = torch.cat(upscaled_maps, dim=1)

        # Apply a 1x1 convolution for dimensionality reduction
        fused_features = self.conv1x1(concatenated_features)

        # Apply self-attention
        refined_features = self.self_attention(fused_features)

        # Use stacked transposed convolutions to upsample to H x W
        for upsampling_layer in self.upsampling_layers:
            refined_features = upsampling_layer(refined_features)

        # Final output with adjusted channels
        segmentation_map = self.adjust_channels(refined_features)

        return segmentation_map


# Example usage
hidden_sizes = [32, 64, 160, 256]
# hidden_sizes = [64, 128, 320, 512]
# hidden_sizes = [64, 128, 320, 512]
input_feature_maps = [torch.randn(1, hidden_sizes[0], 64, 64),   # H/4 * W/4
                      torch.randn(1, hidden_sizes[1], 32, 32),  # H/8 * W/8
                      torch.randn(1, hidden_sizes[2], 16, 16),  # H/16 * W/16
                      torch.randn(1, hidden_sizes[3], 8, 8)]    # H/32 * W/32

# Initialize with all scales
mafm = MultiScaleAttentionFusionModule(in_channels_list=hidden_sizes, out_channels=1)
print(sum(p.numel() for p in mafm.parameters()))
output_segmentation = mafm(input_feature_maps)

print(output_segmentation.shape)  # Expected output: (batch_size, 1, H, W)

