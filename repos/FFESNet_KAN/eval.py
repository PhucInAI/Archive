"""Evaluation code from predicted map"""

# pylint: disable=invalid-name, line-too-long

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

from FFESNet.utils.eval_functions import (
                                            StructureMeasure,
                                            original_WFb,
                                            Fmeasure_calu,
                                            EnhancedMeasure,
                                         )

GT_DIR = '/home/ptn/Storage/Research/FFESNet/data/LA/CVC-ClinicDB/test/masks'
PRED_DIR = '/home/ptn/Storage/Research/FFESNet/runs/LA/CVC-ClinicDB/structure_loss/FFESNet_B4_CSP_mafm/01/predict'

def eval_img_pair(gt_path, pred_path):
    """Evaluation for each image"""
    # --------------------------------------------------------------------
    # Open masks
    # --------------------------------------------------------------------
    pred_mask = np.array(Image.open(gt_path))
    gt_mask = np.array(Image.open(pred_path))

    if len(pred_mask.shape) != 2:
        pred_mask = pred_mask[:, :, 0]
    if len(gt_mask.shape) != 2:
        gt_mask = gt_mask[:, :, 0]

    gt_mask = gt_mask.astype(np.float64) / 255
    gt_mask = (gt_mask > 0.5).astype(np.float64)

    pred_mask = pred_mask.astype(np.float64) / 255
    pred_mask = (pred_mask > 0.5).astype(np.float64)

    # --------------------------------------------------------------------
    # Calculate
    # --------------------------------------------------------------------
    score_smeasure = StructureMeasure(pred_mask, gt_mask)
    score_wFmeasure = original_WFb(pred_mask, gt_mask)
    score_mae = np.mean(np.abs(gt_mask - pred_mask))

    Thresholds = np.linspace(1, 0, 256)
    threshold_E = np.zeros(len(Thresholds))
    threshold_F = np.zeros(len(Thresholds))
    threshold_Pr = np.zeros(len(Thresholds))
    threshold_Rec = np.zeros(len(Thresholds))
    threshold_Iou = np.zeros(len(Thresholds))
    threshold_Spe = np.zeros(len(Thresholds))
    threshold_Dic = np.zeros(len(Thresholds))

    for j, threshold in enumerate(Thresholds):
        threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)
        Bi_pred = np.zeros_like(pred_mask)
        Bi_pred[pred_mask >= threshold] = 1
        threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)

    meanEm = np.mean(threshold_E)
    meanSen = np.mean(threshold_Rec)
    meanSpe = np.mean(threshold_Spe)
    meanDic = np.mean(threshold_Dic)
    meanIoU = np.mean(threshold_Iou)

    return meanDic, meanIoU, meanSpe, meanSen, score_smeasure, meanEm, score_wFmeasure, score_mae


def main():
    """Main function"""
    gt_path_lst = glob.glob(os.path.join(GT_DIR, '*'))

    meanDic, meanIoU, meanSpe, meanSen, score_smeasure, meanEm, score_wFmeasure, score_mae = [], [], [], [], [], [], [], []

    for gt_path in tqdm(gt_path_lst):
        pred_path = os.path.join(PRED_DIR, os.path.basename(gt_path))

        if not os.path.exists(pred_path):
            pred_path = pred_path.split('.')[0] + '.jpg'
        if not os.path.exists(pred_path):
            pred_path = pred_path.split('.')[0] + '.png'

        result = eval_img_pair(gt_path, pred_path)
        meanDic.append(result[0])
        meanIoU.append(result[1])
        meanSpe.append(result[2])
        meanSen.append(result[3])
        score_smeasure.append(result[4])
        meanEm.append(result[5])
        score_wFmeasure.append(result[6])
        score_mae.append(result[7])

    print(f'mDice: {np.mean(meanDic):.4f}')
    print(f'mIoU: {np.mean(meanIoU):.4f}')
    print(f'Precision: {np.mean(meanSpe):.4f}')
    print(f'Recal: {np.mean(meanSen):.4f}')
    print(f'Smeasure: {np.mean(score_smeasure):.4f}')
    print(f'Emeasure: {np.mean(meanEm):.4f}')
    print(f'wFmeasure: {np.mean(score_wFmeasure):.4f}')
    print(f'MAE: {np.mean(score_mae):.4f}')


if __name__=="__main__":
    main()
