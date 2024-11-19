"""
Loss functions
Refefence paper: A survey of loss functions for semantic segmentation
https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions

NOTE: Edit from Tensorflow (original source) to Pytorch 
"""

# pylint: disable=all

import numpy as np
from scipy.ndimage import zoom
from pytorch_msssim import SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLossFunctions:
    """
    A collection of semantic segmentation loss functions for training deep learning models.
    """
    def __init__(self):
        """
        Initialize the SemanticLossFunctions class with pre-defined constants.
        """
        # msg = "Semantic loss functions initialized"
        # aiLogger.info(msg)
        self.epsilon = 1e-5
        self.smooth = 1.0
        self.beta = 0.25
        self.alpha = 0.25
        self.gamma = 2.0


    def dice_coef(self, y_true, y_pred):
        """
        Compute the Dice Coefficient between true and predicted masks.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Dice coefficient value.
        """
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.epsilon) / (
                torch.sum(y_true_f) + torch.sum(y_pred_f) + self.epsilon)

    def sensitivity(self, y_true, y_pred):
        """
        Compute the sensitivity (recall) of the prediction.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Sensitivity value.
        """
        true_positives = torch.sum((y_true * y_pred).round().clamp(0, 1))
        possible_positives = torch.sum(y_true.round().clamp(0, 1))
        return true_positives / (possible_positives + self.epsilon)

    def specificity(self, y_true, y_pred):
        """
        Compute the specificity of the prediction.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Specificity value.
        """
        true_negatives = torch.sum(((1 - y_true) * (1 - y_pred)).round().clamp(0, 1))
        possible_negatives = torch.sum((1 - y_true).round().clamp(0, 1))
        return true_negatives / (possible_negatives + self.epsilon)

    def convert_to_logits(self, y_pred):
        """
        Convert probabilities to logits.

        Args:
            y_pred (torch.Tensor): Predicted probabilities.

        Returns:
            torch.Tensor: Logits.
        """
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)
        return torch.log(y_pred / (1 - y_pred))

    def weighted_cross_entropy_loss(self, y_true, y_pred):
        """
        Compute the weighted binary cross entropy loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted probabilities.

        Returns:
            torch.Tensor: Weighted binary cross entropy loss.
        """
        y_pred = self.convert_to_logits(y_pred)
        pos_weight = torch.tensor(self.beta / (1 - self.beta))
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_weight)
        return loss

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        """
        Compute focal loss given logits and targets.

        Args:
            logits (torch.Tensor): Logits from the model.
            targets (torch.Tensor): Ground truth binary mask.
            alpha (float): Weighting factor for the class.
            gamma (float): Focusing parameter to down-weight easy examples.
            y_pred (torch.Tensor): Predicted probabilities.

        Returns:
            torch.Tensor: Focal loss.
        """
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (F.softplus(-torch.abs(logits)) + torch.relu(-logits)) * (weight_a + weight_b) + logits * weight_b # pylint: disable=not-callable

    def focal_loss(self, y_true, y_pred):
        """
        Compute the focal loss for binary classification.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted probabilities.

        Returns:
            torch.Tensor: Focal loss value.
        """
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)
        logits = self.convert_to_logits(y_pred)
        loss = self.focal_loss_with_logits(logits=logits, targets=y_true, alpha=self.alpha, gamma=self.gamma, y_pred=y_pred)
        return loss.mean()

    def generalized_dice_coefficient(self, y_true, y_pred):
        """
        Compute the generalized Dice coefficient.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Generalized Dice coefficient value.
        """
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + self.smooth) / (
                torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        """
        Compute the Dice loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Dice loss value.
        """
        return 1 - self.generalized_dice_coefficient(y_true, y_pred)

    def bce_dice_loss(self, y_true, y_pred):
        """
        Compute a combination of binary cross-entropy loss and Dice loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted probabilities.

        Returns:
            torch.Tensor: Combined BCE and Dice loss value.
        """
        bce_loss = F.binary_cross_entropy(y_pred, y_true)
        dice_loss = self.dice_loss(y_true, y_pred)
        return (bce_loss + dice_loss) / 2.0

    def confusion(self, y_true, y_pred):
        """
        Compute confusion matrix elements (precision and recall).

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Precision and recall values.
        """
        y_pred_pos = torch.clamp(y_pred, 0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = torch.clamp(y_true, 0, 1)
        y_neg = 1 - y_pos
        tp = torch.sum(y_pos * y_pred_pos)
        fp = torch.sum(y_neg * y_pred_pos)
        fn = torch.sum(y_pos * y_pred_neg)
        prec = (tp + self.smooth) / (tp + fp + self.smooth)
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return prec, recall

    def true_positive(self, y_true, y_pred):
        """
        Compute the true positive rate.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: True positive rate value.
        """
        y_pred_pos = y_pred.round().clamp(0, 1)
        y_pos = y_true.round().clamp(0, 1)
        tp = (torch.sum(y_pos * y_pred_pos) + self.smooth) / (torch.sum(y_pos) + self.smooth)
        return tp

    def true_negative(self, y_true, y_pred):
        """
        Compute the true negative rate.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: True negative rate value.
        """
        y_pred_pos = y_pred.round().clamp(0, 1)
        y_pred_neg = 1 - y_pred_pos
        y_pos = y_true.round().clamp(0, 1)
        y_neg = 1 - y_pos
        tn = (torch.sum(y_neg * y_pred_neg) + self.smooth) / (torch.sum(y_neg) + self.smooth)
        return tn

    def tversky_index(self, y_true, y_pred):
        """
        Compute the Tversky index.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Tversky index value.
        """
        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + self.smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + self.smooth)

    def tversky_loss(self, y_true, y_pred):
        """
        Compute the Tversky loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Tversky loss value.
        """
        return 1 - self.tversky_index(y_true, y_pred)

    def focal_tversky(self, y_true, y_pred):
        """
        Compute the focal Tversky loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Focal Tversky loss value.
        """
        pt_1 = self.tversky_index(y_true, y_pred)
        gamma = 0.75
        return torch.pow((1 - pt_1), gamma)

    def log_cosh_dice_loss(self, y_true, y_pred):
        """
        Compute the logarithm of the hyperbolic cosine of the Dice loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Log-cosh Dice loss value.
        """
        x = self.dice_loss(y_true, y_pred)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def jacard_similarity(self, y_true, y_pred):
        """
        Compute the Jaccard similarity coefficient.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Jaccard similarity coefficient value.
        """
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
        return intersection / union

    def jacard_loss(self, y_true, y_pred):
        """
        Compute the Jaccard loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Jaccard loss value.
        """
        return 1 - self.jacard_similarity(y_true, y_pred)

    def ssim_loss(self, y_true, y_pred):
        """
        Compute the structural similarity index (SSIM) loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: SSIM loss value.
        """
        ssim_module = SSIM(data_range=1, size_average=True, channel=1)
        ssim_loss = 1 - ssim_module(y_true, y_pred)

        return ssim_loss

    def unet3p_hybrid_loss(self, y_true, y_pred):
        """
        Compute the hybrid loss for UNet++ by combining focal loss, SSIM loss, and Jaccard loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Hybrid loss value for UNet++.
        """
        focal_loss = self.focal_loss(y_true, y_pred)
        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)
        return focal_loss + ms_ssim_loss + jacard_loss

    def basnet_hybrid_loss(self, y_true, y_pred):
        """
        Compute the hybrid loss for BASNet by combining BCE loss, SSIM loss, and Jaccard loss.

        Args:
            y_true (torch.Tensor): Ground truth binary mask.
            y_pred (torch.Tensor): Predicted binary mask.

        Returns:
            torch.Tensor: Hybrid loss value for BASNet.
        """
        bce_loss = F.binary_cross_entropy(y_pred, y_true)
        ms_ssim_loss = self.ssim_loss(y_true, y_pred)
        jacard_loss = self.jacard_loss(y_true, y_pred)
        return bce_loss + ms_ssim_loss + jacard_loss


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

# ########################################################################
# Adaptive tvMF Dice loss
# ########################################################################
class Adaptive_tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(Adaptive_tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        # score = F.normalize(score, p=2, dim=[0,1,2])
        # target = F.normalize(target, p=2, dim=[0,1,2])
        score = score.view(-1)
        target = target.view(-1)
        cosine = torch.sum(score * target)/(torch.sqrt(torch.sum(score*score))*torch.sqrt(torch.sum(target*target)))
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, kappa=None, sigmoid=False):
        if sigmoid:
            inputs = torch.sigmoid(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0

        for i in range(0, self.n_classes):
            # print(inputs[:, i].size())
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], kappa[i])
            loss += tvmf_dice
        return loss / self.n_classes