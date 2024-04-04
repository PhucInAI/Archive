import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

from FFESNet.models.FFESNet import FFESNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FFESNet('B0', pred_type='AA')
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