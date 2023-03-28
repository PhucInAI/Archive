##########################################################################
# Pure Pytorch correlation layer
# Refactor      :   Phuc Thanh Nguyen
# Created       :   Mar 24th, 2023
# Last edited   :   Mar 27th, 2023
# References    :   https://github.com/limacv/CorrelationLayer
##########################################################################

import  torch
import  torch.nn as nn

class   CorrelationFunction(nn.Module):
    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        assert kernel_size  ==  1, "kernel_size other than 1 is not implemented"
        assert pad_size     ==  max_displacement
        assert stride1      ==  stride2 == 1
        
        super().__init__()
        
        self.max_hdisp      =   max_displacement
        self.padlayer       =   nn.ConstantPad2d(pad_size, 0)


    def forward(self, in1, in2):
        in2_pad             =   self.padlayer(in2)
        offsety, offsetx    =   torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                           torch.arange(0, 2 * self.max_hdisp + 1)])
        hei, wid            =   in1.shape[2], in1.shape[3]
        
        output              = torch.cat([
                                            torch.mean(in1 * in2_pad[:, :, dy:dy+hei, dx:dx+wid], 1, keepdim=True)
                                            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
                                        ],
                                        1)
        
        return output