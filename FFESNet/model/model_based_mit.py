"""Model define"""


import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

import FFESNet.model.segformer.mit as mit
import FFESNet.model.segformer.mlp as mlp
from FFESNet.model.auxiliary_components import (
                                                CBLinear,
                                                CBFuse,
                                                Conv,
                                                RepNCSPELAN4,
                                                ADown,
                                               )
from FFESNet.utils.ai_logger import aiLogger


class ModelBasedMit(nn.Module):
    """Model defined based on MixTransformer"""
    def __init__(self, model_backbone_type = 'B0', feature_fuse='LP', prediction='linear', mode='inferene'):
        """Init function"""
        super().__init__()
        self.model_backbone_type = model_backbone_type
        self.mode = mode


        # ################################################################
        # Main branch
        # ################################################################


        # ----------------------------------------------------------------
        # Feature extractor
        # ----------------------------------------------------------------
        if self.model_backbone_type == 'B0':
            self.backbone = mit.mit_b0()
        if self.model_backbone_type == 'B1':
            self.backbone = mit.mit_b1()
        if self.model_backbone_type == 'B2':
            self.backbone = mit.mit_b2()
        if self.model_backbone_type == 'B3':
            self.backbone = mit.mit_b3()
        if self.model_backbone_type == 'B4':
            self.backbone = mit.mit_b4()
        if self.model_backbone_type == 'B5':
            self.backbone = mit.mit_b5()

        self._init_weights()  # load pretrain

        # ----------------------------------------------------------------
        # Header
        # ----------------------------------------------------------------
        if feature_fuse == "LP":
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
        else:
            self.ff_1 = RepNCSPELAN4(
                                        self.backbone.embed_dims[0],
                                        self.backbone.embed_dims[0],
                                        self.backbone.embed_dims[0]//2,
                                        self.backbone.embed_dims[0]//4,
                                        1
                                    )
            self.ff_2 = RepNCSPELAN4(
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[1]//2,
                                        self.backbone.embed_dims[0]//4,
                                        1
                                    )
            self.ff_3 = RepNCSPELAN4(
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[2]//2,
                                        self.backbone.embed_dims[0]//4,
                                        1
                                    )
            self.ff_4 = RepNCSPELAN4(
                                        self.backbone.embed_dims[3],
                                        self.backbone.embed_dims[3],
                                        self.backbone.embed_dims[3]//2,
                                        self.backbone.embed_dims[0]//4,
                                        1
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
        if feature_fuse == 'LP':
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
        else:
            self.ff_12 = RepNCSPELAN4(
                                        self.backbone.embed_dims[0],
                                        self.backbone.embed_dims[0],
                                        self.backbone.embed_dims[0]//2,
                                        self.backbone.embed_dims[0]//4,
                                        1
                                     )
            self.ff_23 = RepNCSPELAN4(
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[1]//2,
                                        self.backbone.embed_dims[1]//4,
                                        1
                                     )
            self.ff_34 = RepNCSPELAN4(
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[2]//2,
                                        self.backbone.embed_dims[2]//4,
                                        1
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

        if self.mode == 'train':

            # ################################################################
            # Auxiliary branch
            # ################################################################


            # ----------------------------------------------------------------
            # Rout from main branch
            # ----------------------------------------------------------------
            self.rout1 = CBLinear(
                                    self.backbone.embed_dims[1],
                                    [
                                        self.backbone.embed_dims[1]
                                    ]
                                )
            self.rout2 = CBLinear(
                                    self.backbone.embed_dims[2],
                                    [
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[2]
                                    ]
                                )
            self.rout3 = CBLinear(
                                    self.backbone.embed_dims[3],
                                    [
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[3]
                                    ]
                                )

            # ----------------------------------------------------------------
            # Fast downscale from input
            # ----------------------------------------------------------------
            self.aux_conv1 = Conv(3, self.backbone.embed_dims[0], 3, 2)
            self.aux_conv2 = Conv(self.backbone.embed_dims[0], self.backbone.embed_dims[1], 3, 2)

            # ----------------------------------------------------------------
            # ELAN blocks
            # ----------------------------------------------------------------
            self.elan1 = RepNCSPELAN4(
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[0],
                                        self.backbone.embed_dims[0]//2,
                                        1
                                    )
            self.adown1 = ADown(self.backbone.embed_dims[1], self.backbone.embed_dims[1])
            self.cbfuse1 = CBFuse([0, 0, 0])

            self.elan2 = RepNCSPELAN4(
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[1],
                                        self.backbone.embed_dims[0],
                                        1
                                    )
            self.adown2 = ADown(self.backbone.embed_dims[2], self.backbone.embed_dims[2])
            self.cbfuse2 = CBFuse([1, 1])

            self.elan3 = RepNCSPELAN4(
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[3],
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[1],
                                        1
                                    )
            self.adown3 = ADown(self.backbone.embed_dims[3], self.backbone.embed_dims[3])
            self.cbfuse3 = CBFuse([2])

            self.elan4 = RepNCSPELAN4(
                                        self.backbone.embed_dims[3],
                                        self.backbone.embed_dims[3],
                                        self.backbone.embed_dims[2],
                                        self.backbone.embed_dims[1],
                                        1
                                    )

            # ----------------------------------------------------------------
            # Prediction of auxiliary branch
            # ----------------------------------------------------------------
            self.linear_pred_aux = nn.Conv2d(
                                                self.backbone.embed_dims[1] +\
                                                self.backbone.embed_dims[2] +\
                                                self.backbone.embed_dims[3]*2,
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


        # ################################################################
        # Main branch
        # ################################################################


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


        # ################################################################
        # Auxiliary branch
        # ################################################################

        if self.mode == 'train':
            aux_rout1 = self.rout1(out_2)
            aux_rout2 = self.rout2(out_3)
            aux_rout3 = self.rout3(out_4)

            aux_out = self.aux_conv1(x)
            aux_out = self.aux_conv2(aux_out)

            aux_out = self.elan1(aux_out)
            aux_out1 = aux_out
            aux_out = self.adown1(aux_out)
            aux_out = self.cbfuse1([aux_rout1, aux_rout2, aux_rout3, aux_out])

            aux_out = self.elan2(aux_out)
            aux_out2 = aux_out
            aux_out = self.adown2(aux_out)
            aux_out = self.cbfuse2([aux_rout2, aux_rout3, aux_out])

            aux_out = self.elan3(aux_out)
            aux_out3 = aux_out
            aux_out = self.adown3(aux_out)
            aux_out = self.cbfuse3([aux_rout3, aux_out])

            aux_out = self.elan4(aux_out)
            aux_out4 = aux_out

            aux_out4 = F.interpolate(aux_out4,scale_factor=8,mode='bilinear', align_corners=False)
            aux_out3 = F.interpolate(aux_out3,scale_factor=4,mode='bilinear', align_corners=False)
            aux_out2 = F.interpolate(aux_out2,scale_factor=2,mode='bilinear', align_corners=False)
            aux_out1 = F.interpolate(aux_out1,scale_factor=1,mode='bilinear', align_corners=False)

            aux_out = self.linear_pred_aux(torch.cat([aux_out1, aux_out2, aux_out3, aux_out4], dim=1))
            aux_out = F.interpolate(aux_out, scale_factor=4, mode='bilinear', align_corners=False)

            return out, aux_out

        else:
            return out


def main():
    """Test function"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModelBasedMit('B0')
    model.to(device)

    # --------------------------------------------------------------------
    # Feedforward
    # --------------------------------------------------------------------
    x = torch.rand(1,3,352,352).to(device)
    y_pred, y_pred_aux = model(x)
    print('Successfully feedfoward!!!')

    # --------------------------------------------------------------------
    # Backprop
    # --------------------------------------------------------------------
    y = torch.rand(1,1,352,352).to(device)
    y_pred = F.upsample(y_pred, size=y.shape[2:], mode='bilinear', align_corners=False)
    y_pred_aux = F.upsample(y_pred_aux, size=y.shape[2:], mode='bilinear', align_corners=False)
    y_pred = y_pred.sigmoid()
    y_pred_aux = y_pred_aux.sigmoid()
    loss_main = F.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')
    loss_aux = F.binary_cross_entropy_with_logits(y_pred_aux, y, reduction='mean')

    loss = loss_main + 0.5*loss_aux

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Successfully backprop!!!')


if __name__=="__main__":
    main()
