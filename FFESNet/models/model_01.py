"""Model define"""

# pylint: disable=no-member
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

import FFESNet.models.mit as mit
import FFESNet.models.mlp as mlp


class Model(nn.Module):
    """Model define"""
    def __init__(self, model_type = 'B0'):
        """Init function"""
        super().__init__()
        self.model_type = model_type

        # ----------------------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------------------
        if self.model_type == 'B0':
            self.backbone = mit.mit_b0()
        if self.model_type == 'B1':
            self.backbone = mit.mit_b1()
        if self.model_type == 'B2':
            self.backbone = mit.mit_b2()
        if self.model_type == 'B3':
            self.backbone = mit.mit_b3()
        if self.model_type == 'B4':
            self.backbone = mit.mit_b4()
        if self.model_type == 'B5':
            self.backbone = mit.mit_b5()

        self._init_weights()  # load pretrain

        # ----------------------------------------------------------------
        # LP Header
        # ----------------------------------------------------------------
        self.lp_1 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.lp_2 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.lp_3 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])
        self.lp_4 = mlp.LP(input_dim = self.backbone.embed_dims[3], embed_dim = self.backbone.embed_dims[3])

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
        # Fused LP Header
        # ----------------------------------------------------------------
        self.lp_12 = mlp.LP(
                            input_dim = self.backbone.embed_dims[0],
                            embed_dim = self.backbone.embed_dims[0]
                           )
        self.lp_23 = mlp.LP(
                            input_dim = self.backbone.embed_dims[1],
                            embed_dim = self.backbone.embed_dims[1]
                           )
        self.lp_34 = mlp.LP(
                            input_dim = self.backbone.embed_dims[2],
                            embed_dim = self.backbone.embed_dims[2]
                           )

        # ----------------------------------------------------------------
        # Final Linear Prediction
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
        """
        Init pretrained
        """
        if self.model_type == 'B0':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/models/pretrained/mit_b0.pth')
        if self.model_type == 'B1':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/models/pretrained/mit_b1.pth')
        if self.model_type == 'B2':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/models/pretrained/mit_b2.pth')
        if self.model_type == 'B3':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/models/pretrained/mit_b3.pth')
        if self.model_type == 'B4':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/models/pretrained/mit_b4.pth')
        if self.model_type == 'B5':
            pretrained_dict = torch.load('/home/ptn/Storage/Research/FFESNet/FFESNet/models/pretrained/mit_b5.pth')

        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("Successfully loaded pretrained!!!!")


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

        # Go through LP Header
        lp_1 = self.lp_1(out_1)
        lp_2 = self.lp_2(out_2)
        lp_3 = self.lp_3(out_3)
        lp_4 = self.lp_4(out_4)

        # Linear fuse and go pass LP Header
        lp_34 = self.lp_34(self.linear_fuse34(torch.cat([
                                                            lp_3,
                                                            F.interpolate(
                                                                            lp_4,
                                                                            scale_factor=2,
                                                                            mode='bilinear',
                                                                            align_corners=False
                                                                          )
                                                        ],
                                                        dim=1
                                                       )))
        lp_23 = self.lp_23(self.linear_fuse23(torch.cat([
                                                            lp_2,
                                                            F.interpolate(
                                                                            lp_34,
                                                                            scale_factor=2,
                                                                            mode='bilinear',
                                                                            align_corners=False
                                                                         )
                                                        ],
                                                        dim=1
                                                       )))
        lp_12 = self.lp_12(self.linear_fuse12(torch.cat([
                                                            lp_1,
                                                            F.interpolate(
                                                                            lp_23,
                                                                            scale_factor=2,
                                                                            mode='bilinear',
                                                                            align_corners=False
                                                                         )
                                                        ],
                                                        dim=1
                                                       )))

        # Get the final output
        lp4_resized = F.interpolate(lp_4,scale_factor=8,mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34,scale_factor=4,mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)
        lp1_resized = lp_12

        out = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

        return out


def main():
    """Test function"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model('B0')
    model.to(device)

    # --------------------------------------------------------------------
    # Feedforward
    # --------------------------------------------------------------------
    x = torch.rand(1,3,352,352).to(device)
    y_pred = model(x)
    print('Successfully feedfoward!!!')

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
