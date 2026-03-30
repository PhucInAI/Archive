"""Model define"""


# pylint: disable=C0103

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

import FFESNet.model.segformer.mit as mit
import FFESNet.model.segformer.mlp as mlp
# from fvcore.nn.flop_count import flop_count
from FFESNet.utils.ai_logger import aiLogger


class ModelBasedMit(nn.Module):
    """Model defined based on MixTransformer"""
    def __init__(self, model_backbone_type = 'B0', use_KAN = True, prediction='linear', mode='inferene'):
        """Init function"""
        super().__init__()
        self.model_backbone_type = model_backbone_type
        self.use_KAN = use_KAN
        self.mode = mode
        self.prediction = prediction

        # ----------------------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------------------
        if use_KAN:
            act_layer ="KAT"
        else:
            act_layer = nn.GELU
        if self.model_backbone_type == 'B0':
            self.backbone = mit.mit_b0(act_layer)
        if self.model_backbone_type == 'B1':
            self.backbone = mit.mit_b1(act_layer)
        if self.model_backbone_type == 'B2':
            self.backbone = mit.mit_b2(act_layer)
        if self.model_backbone_type == 'B3':
            self.backbone = mit.mit_b3(act_layer)
        if self.model_backbone_type == 'B4':
            self.backbone = mit.mit_b4(act_layer)
        if self.model_backbone_type == 'B5':
            self.backbone = mit.mit_b5(act_layer)

        if not self.use_KAN:
            self._init_weights()  # load pretrain

        # ----------------------------------------------------------------
        # Header
        # ----------------------------------------------------------------
        self.ff_1 = mlp.LP(
                            input_dim = self.backbone.embed_dims[0],
                            embed_dim = self.backbone.embed_dims[0]
                            )
        self.ff_2 = mlp.LP(
                            input_dim = self.backbone.embed_dims[1],
                            embed_dim = self.backbone.embed_dims[1]
                            )
        self.ff_3 = mlp.LP(
                            input_dim = self.backbone.embed_dims[2],
                            embed_dim = self.backbone.embed_dims[2]
                            )
        self.ff_4 = mlp.LP(
                            input_dim = self.backbone.embed_dims[3],
                            embed_dim = self.backbone.embed_dims[3]
                            )

        # ----------------------------------------------------------------
        # Linear Fuse
        # ----------------------------------------------------------------
        self.linear_fuse34 = ConvModule(
                                        in_channels=(self.backbone.embed_dims[2] + self.backbone.embed_dims[3]),
                                        out_channels=self.backbone.embed_dims[2],
                                        kernel_size=1,
                                        norm_cfg={'type': 'BN', 'requires_grad':True}
                                       )
        self.linear_fuse23 = ConvModule(
                                        in_channels=(self.backbone.embed_dims[1] + self.backbone.embed_dims[2]),
                                        out_channels=self.backbone.embed_dims[1],
                                        kernel_size=1,
                                        norm_cfg={'type':'BN', 'requires_grad':True}
                                       )
        self.linear_fuse12 = ConvModule(
                                        in_channels=(self.backbone.embed_dims[0] + self.backbone.embed_dims[1]),
                                        out_channels=self.backbone.embed_dims[0],
                                        kernel_size=1,
                                        norm_cfg={'type':'BN', 'requires_grad':True}
                                       )

        # ----------------------------------------------------------------
        # Fused Header
        # ----------------------------------------------------------------
        self.ff_12 = mlp.LP(
                            input_dim = self.backbone.embed_dims[0],
                            embed_dim = self.backbone.embed_dims[0]
                            )
        self.ff_23 = mlp.LP(
                            input_dim = self.backbone.embed_dims[1],
                            embed_dim = self.backbone.embed_dims[1]
                            )
        self.ff_34 = mlp.LP(
                            input_dim = self.backbone.embed_dims[2],
                            embed_dim = self.backbone.embed_dims[2]
                            )

        # ----------------------------------------------------------------
        # Final Prediction
        # ----------------------------------------------------------------
        self.linear_pred = nn.Conv2d(
                                        (
                                            self.backbone.embed_dims[0] +\
                                            self.backbone.embed_dims[1] +\
                                            self.backbone.embed_dims[2] +\
                                            self.backbone.embed_dims[3]
                                        ),
                                        1,
                                        kernel_size=1
                                    )


    def _init_weights(self):
        """Init pretrained"""
        if self.model_backbone_type == 'B0':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/model/pretrained/mit_b0.pth')
        if self.model_backbone_type == 'B1':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/model/pretrained/mit_b1.pth')
        if self.model_backbone_type == 'B2':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/model/pretrained/mit_b2.pth')
        if self.model_backbone_type == 'B3':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/model/pretrained/mit_b3.pth')
        if self.model_backbone_type == 'B4':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/model/pretrained/mit_b4.pth')
        if self.model_backbone_type == 'B5':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/model/pretrained/mit_b5.pth')

        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)

        msg = "Successfully loaded pretrained!!!!"
        aiLogger.info(msg)


    def forward(self, x):
        """Foward function"""
        # ----------------------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------------------
        b = x.shape[0]

        # Stage 1
        out_1, h, w = self.backbone.patch_embed1(x)
        for _, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, h, w)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[0], 88, 88)

        # Stage 2
        out_2, h, w = self.backbone.patch_embed2(out_1)
        for _, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, h, w)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[1], 44, 44)

        # Stage 3
        out_3, h, w = self.backbone.patch_embed3(out_2)
        for _, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, h, w)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[2], 22, 22)

        # Stage 4
        out_4, h, w = self.backbone.patch_embed4(out_3)
        for _, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, h, w)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[3], 11, 11)

        # ----------------------------------------------------------------
        # Fusion feature map
        # ----------------------------------------------------------------
        # Go through Fuse Header
        ff_1 = self.ff_1(out_1)
        ff_2 = self.ff_2(out_2)
        ff_3 = self.ff_3(out_3)
        ff_4 = self.ff_4(out_4)

        # Linear fuse and go pass Fused Header
        ff_34 = self.ff_34(self.linear_fuse34(torch.cat([
                                                            ff_3,
                                                            F.interpolate(
                                                                            ff_4,
                                                                            scale_factor=2,
                                                                            mode='bilinear',
                                                                            align_corners=False
                                                                          )
                                                        ],
                                                        dim=1
                                                       )))
        ff_23 = self.ff_23(self.linear_fuse23(torch.cat([
                                                            ff_2,
                                                            F.interpolate(
                                                                            ff_34,
                                                                            scale_factor=2,
                                                                            mode='bilinear',
                                                                            align_corners=False
                                                                         )
                                                        ],
                                                        dim=1
                                                       )))
        ff_12 = self.ff_12(self.linear_fuse12(torch.cat([
                                                            ff_1,
                                                            F.interpolate(
                                                                            ff_23,
                                                                            scale_factor=2,
                                                                            mode='bilinear',
                                                                            align_corners=False
                                                                         )
                                                        ],
                                                        dim=1
                                                       )))

        # ----------------------------------------------------------------
        # Get the final output
        # ----------------------------------------------------------------
        ff_4_resized = F.interpolate(ff_4,scale_factor=8,mode='bilinear', align_corners=False)
        ff_3_resized = F.interpolate(ff_34,scale_factor=4,mode='bilinear', align_corners=False)
        ff_2_resized = F.interpolate(ff_23,scale_factor=2,mode='bilinear', align_corners=False)
        ff_1_resized = ff_12

        out = self.linear_pred(torch.cat([ff_1_resized, ff_2_resized, ff_3_resized, ff_4_resized], dim=1))
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out


def main():
    """Test function"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModelBasedMit('B0', use_KAN=True)
    model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6)


    # --------------------------------------------------------------------
    # Feedforward
    # --------------------------------------------------------------------
    x = torch.rand(1,3,352,352).to(device)
    y_pred = model(x)
    # print('Successfully feedfoward!!!')

    # gflop_dict, _ = flop_count(model, x)
    # gflops = sum(gflop_dict.values())
    # print("GFLOPs:", gflops)
    # y_pred, y_pred_aux = model(x)
    # print('Successfully feedfoward!!!')

    # from calflops import calculate_flops
    # from torchvision import models

    # batch_size = 1
    # input_shape = (batch_size, 3, 352, 352)
    # flops, macs, params = calculate_flops(model=model,
    #                                     input_shape=input_shape,
    #                                     output_as_string=True,
    #                                     output_precision=4)
    # print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    # --------------------------------------------------------------------
    # Backprop
    # --------------------------------------------------------------------
    y = torch.rand(1,1,352,352).to(device)
    y_pred = F.upsample(y_pred, size=y.shape[2:], mode='bilinear', align_corners=False)
    y_pred = y_pred.sigmoid()
    loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Successfully backprop!!!')


if __name__=="__main__":
    main()
