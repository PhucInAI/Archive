import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class GatedSEMSCAM(nn.Module):
    def __init__(self, channels, reduction=2):
        super(GatedSEMSCAM, self).__init__()
        
        # Global context path (SE enhancement with fully connected layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)  # FC layer 1
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)  # FC layer 2

        # Local context path (X2 path with pointwise convolutions)
        self.conv1_x2 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.bn1_x2 = nn.BatchNorm2d(channels // reduction)
        self.conv2_x2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.bn2_x2 = nn.BatchNorm2d(channels)
        
        # Gated attention mechanism
        self.gate_x1 = nn.Parameter(torch.tensor(0.5))  # Learnable gate for global context (X1)
        self.gate_x2 = nn.Parameter(torch.tensor(0.5))  # Learnable gate for local context (X2)

    def forward(self, x):
        # Global context path (X1 with SE-like fully connected layers)
        x1 = self.global_avg_pool(x).view(x.size(0), -1)  # Global avg pool (C, 1, 1) -> reshape to (C)
        x1 = self.fc1(x1)                                 # Fully connected (C/r)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)                                 # Fully connected (C)
        x1 = torch.sigmoid(x1).view(x.size(0), -1, 1, 1)  # Sigmoid, reshape to (C, 1, 1)

        # Local context path (X2 with pointwise convs)
        x2 = self.conv1_x2(x)                             # Pointwise conv (C/r, H, W)
        x2 = self.bn1_x2(x2)                              # Batch normalization
        x2 = self.relu(x2)                                # ReLU activation
        x2 = self.conv2_x2(x2)                            # Pointwise conv (C, H, W)
        x2 = self.bn2_x2(x2)                              # Batch normalization

        # Gated combination of X1 and X2
        x1 = x1.expand_as(x2)                             # Expand X1 to match the spatial dimensions of X2 (C, H, W)
        x3 = self.gate_x1 * x1 + self.gate_x2 * x2        # Gated combination of X1 and X2
        x3 = torch.sigmoid(x3)                            # Apply sigmoid to the combination

        # Element-wise multiplication of the input and the attention map
        out = x * x3                                      # Element-wise multiplication with the input X

        return out

class CrossAttention(nn.Module):
    def __init__(self, C_x, C_y):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(C_x, C_x // 8, kernel_size=1)  # Query from X
        self.key_conv = nn.Conv2d(C_y, C_x // 8, kernel_size=1)    # Key from Y
        self.value_conv = nn.Conv2d(C_y, C_x, kernel_size=1)       # Value from Y
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable parameter to scale the attention output

    def forward(self, X, Y):
        Q = self.query_conv(X)  # Query from X
        K = self.key_conv(Y)    # Key from Y
        V = self.value_conv(Y)  # Value from Y

        Q_flat = Q.view(Q.size(0), Q.size(1), -1)  # (batch_size, C_x//8, H*W)
        K_flat = K.view(K.size(0), K.size(1), -1)  # (batch_size, C_x//8, H*W)
        V_flat = V.view(V.size(0), V.size(1), -1)  # (batch_size, C_x, H*W)

        attention = torch.bmm(Q_flat.permute(0, 2, 1), K_flat)  # (batch_size, H*W, H*W)
        attention = F.softmax(attention, dim=-1)  # Softmax along spatial dimensions

        out = torch.bmm(V_flat, attention.permute(0, 2, 1))  # (batch_size, C_x, H*W)
        out = out.view_as(X)  # Reshape to original dimensions (batch_size, C_x, H, W)

        out = self.gamma * out + X

        return out

class DualScaleAttentionFusionModule(nn.Module):
    def __init__(self, C_x, C_y, reduction_ratio=2):
        super(DualScaleAttentionFusionModule, self).__init__()
        # Deformable convolution to align Y with X spatially
        self.deform_conv = DeformConv2d(C_y, C_x, kernel_size=3, padding=1)
        
        # Offsets for the deformable convolution
        self.offset_conv = nn.Conv2d(C_y, 18, kernel_size=3, padding=1)  # 18 = 2 * kernel_size * kernel_size

        # Cross attention between X and Y (aligned)
        self.cross_attention = CrossAttention(C_x, C_y)

        # Gated attention (using the existing GatedSEMSCAM module)
        self.gated_attention = GatedSEMSCAM(C_x, reduction=reduction_ratio)

    def forward(self, X, Y):
        # 1. Upsample Y to match the spatial size of X
        Y_upsampled = F.interpolate(Y, size=X.shape[2:], mode='bilinear', align_corners=False)

        # 2. Calculate offsets for the deformable convolution
        # print('Y_upsampled', Y_upsampled.shape, X.shape, Y.shape)
        offsets = self.offset_conv(Y_upsampled)

        # 3. Apply deformable convolution to better align Y to X
        Y_aligned = self.deform_conv(Y_upsampled, offsets)

        # 4. Apply cross-attention between X and the aligned Y
        # # print(X.shape)
        # # print(Y_aligned.shape)
        X_attended = self.cross_attention(X, Y)

        # 5. Gated attention on the sum of X and Y_aligned
        Z = X_attended + Y_aligned
        # print('Z', Z.shape)
        Z = self.gated_attention(Z)
        # print('Z', Z.shape)

        # 6. Element-wise fusion with learned attention
        Z_final = X * Z + Y_aligned * (1 - Z)
        # print('Z_final', Z_final.shape)

        return Z_final

class MultiScaleAttentionFusionModule(nn.Module):
    def __init__(self, C1, C2, C3, C4):
        super(MultiScaleAttentionFusionModule, self).__init__()
        self.fusion_34 = DualScaleAttentionFusionModule(C3, C4)  # Combine X4 and X3
        self.fusion_23 = DualScaleAttentionFusionModule(C2, C3)  # Combine new X3 and X2
        self.fusion_12 = DualScaleAttentionFusionModule(C1, C2)  # Combine new X2 and X1
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(C1, C1, kernel_size=4, stride=2, padding=1),  # Upsample to C1 * H/2 * W/2
            nn.ConvTranspose2d(C1, C1, kernel_size=4, stride=2, padding=1),  # Upsample to C1 * H/2 * W/2
            nn.Conv2d(C1, C1, kernel_size=3, padding=1),  # Final Conv to maintain channels
        )
        
        # Final convolution to produce the segmentation map
        self.segmentation_conv = nn.Conv2d(C1, 1, kernel_size=1)  # Output (1, H, W)

    def forward(self, X1, X2, X3, X4):
        # Step 1: Combine X4 and X3 to new X3
        new_X3 = self.fusion_34(X3, X4)

        # print(new_X3.shape)
        # print(X2.shape)
        
        # Step 2: Combine new X3 and X2 to new X2
        new_X2 = self.fusion_23(X2, new_X3)
        
        # Step 3: Combine new X2 and X1 to new X1
        new_X1 = self.fusion_12(X1, new_X2)
        
        # Step 4: Upsample new X1 to C1 * H * W
        upsampled_X1 = self.upsample(new_X1)
        
        # Step 5: Final segmentation output
        segmentation_map = self.segmentation_conv(upsampled_X1)
        
        return segmentation_map

# Example usage
C1, C2, C3, C4 = 64, 128, 320, 512  # Define the channel sizes
model = MultiScaleAttentionFusionModule(C1, C2, C3, C4)
# print(sum(p.numel() for p in model.parameters()))

# Test input tensors
X1 = torch.randn(1, C1, 88, 88)  # X1 (C1, H/4, W/4)
X2 = torch.randn(1, C2, 44, 44)  # X2 (C2, H/8, W/8)
X3 = torch.randn(1, C3, 22, 22)  # X3 (C3, H/16, W/16)
X4 = torch.randn(1, C4, 11, 11)  # X4 (C4, H/32, W/32)

output = model(X1, X2, X3, X4)
# print(output.shape)  # Output shape should be (1, 1, H, W)

