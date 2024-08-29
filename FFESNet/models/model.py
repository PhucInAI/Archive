"""Model define"""

# pylint: disable=no-member
import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

import FFESNet.models.mit as mit
import FFESNet.models.mlp as mlp
from FFESNet.models.auxiliary_components import CBLinear, CBFuse, Conv, RepNCSPELAN4, ADown

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
        # self.lp_1 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        # self.lp_2 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        # self.lp_3 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])
        # self.lp_4 = mlp.LP(input_dim = self.backbone.embed_dims[3], embed_dim = self.backbone.embed_dims[3])
        self.lp_1 = RepNCSPELAN4(self.backbone.embed_dims[0], self.backbone.embed_dims[0], self.backbone.embed_dims[0]//2, self.backbone.embed_dims[0]//4, 1)
        self.lp_2 = RepNCSPELAN4(self.backbone.embed_dims[1], self.backbone.embed_dims[1], self.backbone.embed_dims[1]//2, self.backbone.embed_dims[0]//4, 1)
        self.lp_3 = RepNCSPELAN4(self.backbone.embed_dims[2], self.backbone.embed_dims[2], self.backbone.embed_dims[2]//2, self.backbone.embed_dims[0]//4, 1)
        self.lp_4 = RepNCSPELAN4(self.backbone.embed_dims[3], self.backbone.embed_dims[3], self.backbone.embed_dims[3]//2, self.backbone.embed_dims[0]//4, 1)

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
        # self.lp_12 = mlp.LP(
        #                     input_dim = self.backbone.embed_dims[0],
        #                     embed_dim = self.backbone.embed_dims[0]
        #                    )
        # self.lp_23 = mlp.LP(
        #                     input_dim = self.backbone.embed_dims[1],
        #                     embed_dim = self.backbone.embed_dims[1]
        #                    )
        # self.lp_34 = mlp.LP(
        #                     input_dim = self.backbone.embed_dims[2],
        #                     embed_dim = self.backbone.embed_dims[2]
        #                    )
        self.lp_12 = RepNCSPELAN4(self.backbone.embed_dims[0], self.backbone.embed_dims[0], self.backbone.embed_dims[0]//2, self.backbone.embed_dims[0]//4, 1)
        self.lp_23 = RepNCSPELAN4(self.backbone.embed_dims[1], self.backbone.embed_dims[1], self.backbone.embed_dims[1]//2, self.backbone.embed_dims[1]//4, 1)
        self.lp_34 = RepNCSPELAN4(self.backbone.embed_dims[2], self.backbone.embed_dims[2], self.backbone.embed_dims[2]//2, self.backbone.embed_dims[2]//4, 1)

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

        # ----------------------------------------------------------------
        # Auxiliary branch
        # ----------------------------------------------------------------
        self.rout1 = CBLinear(self.backbone.embed_dims[1], [self.backbone.embed_dims[1]])
        self.rout2 = CBLinear(self.backbone.embed_dims[2], [self.backbone.embed_dims[1], self.backbone.embed_dims[2]])
        self.rout3 = CBLinear(self.backbone.embed_dims[3], [self.backbone.embed_dims[1], self.backbone.embed_dims[2], self.backbone.embed_dims[3]])

        self.aux_conv1 = Conv(3, self.backbone.embed_dims[0], 3, 2)
        self.aux_conv2 = Conv(self.backbone.embed_dims[0], self.backbone.embed_dims[1], 3, 2)

        self.elan1 = RepNCSPELAN4(self.backbone.embed_dims[1], self.backbone.embed_dims[1],  self.backbone.embed_dims[0],  self.backbone.embed_dims[0]//2, 1)
        self.adown1 = ADown(self.backbone.embed_dims[1], self.backbone.embed_dims[1])
        self.cbfuse1 = CBFuse([0, 0, 0])

        self.elan2 = RepNCSPELAN4(self.backbone.embed_dims[1], self.backbone.embed_dims[2], self.backbone.embed_dims[1], self.backbone.embed_dims[0], 1)
        self.adown2 = ADown(self.backbone.embed_dims[2], self.backbone.embed_dims[2])
        self.cbfuse2 = CBFuse([1, 1])

        self.elan3 = RepNCSPELAN4(self.backbone.embed_dims[2], self.backbone.embed_dims[3], self.backbone.embed_dims[2], self.backbone.embed_dims[1], 1)
        self.adown3 = ADown(self.backbone.embed_dims[3], self.backbone.embed_dims[3])
        self.cbfuse3 = CBFuse([2])

        self.elan4 = RepNCSPELAN4(self.backbone.embed_dims[3], self.backbone.embed_dims[3], self.backbone.embed_dims[2], self.backbone.embed_dims[1], 1)

        self.linear_pred_aux = nn.Conv2d(
                                        self.backbone.embed_dims[1] + self.backbone.embed_dims[2] + self.backbone.embed_dims[3]*2,
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


        # ----------------------------------------------------------------
        # Auxiliary branch
        # ----------------------------------------------------------------
        # print(self.backbone.embed_dims)
        # print(out_2.shape)
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

        # print(out.shape)
        # print(aux_out1.shape, aux_out2.shape, aux_out3.shape, aux_out4.shape)

        aux_out4 = F.interpolate(aux_out4,scale_factor=8,mode='bilinear', align_corners=False)
        aux_out3 = F.interpolate(aux_out3,scale_factor=4,mode='bilinear', align_corners=False)
        aux_out2 = F.interpolate(aux_out2,scale_factor=2,mode='bilinear', align_corners=False)
        aux_out1 = F.interpolate(aux_out1,scale_factor=1,mode='bilinear', align_corners=False)

        # print(aux_out1.shape, aux_out2.shape, aux_out3.shape, aux_out4.shape)

        aux_out = self.linear_pred_aux(torch.cat([aux_out1, aux_out2, aux_out3, aux_out4], dim=1))
        aux_out = F.interpolate(aux_out, scale_factor=4, mode='bilinear', align_corners=False)

        return out, aux_out


def main():
    """Test function"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model('B0')
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
