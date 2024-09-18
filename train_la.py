"""Train Learning Ability"""


import os
import glob
import shutil
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import torch
# torch.backends.cudnn.enabled = False
import torch.nn.functional as F
from torch.utils.data import DataLoader
import imageio
from skimage import img_as_ubyte

import matplotlib.pyplot as plt

from FFESNet.model.model_based_mit import ModelBasedMit
from FFESNet.utils.dataset import PolypDataset, TestDataset
from FFESNet.utils.loss import SemanticLossFunctions
from FFESNet.utils.metric import calculate_image_pair
from FFESNet.utils.ai_logger import aiLogger

import warnings
warnings.filterwarnings("ignore")


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
global DEVICE # pylint: disable=W0604
DATASET_LST = ['Kvasir-SEG', 'CVC-ColonDB', 'CVC-ClinicDB', 'ETIS-LaribPolypDB']
LOSS_LST =  [
                # 'weighted_cross_entropy_loss',
                # 'focal_loss',
                # 'dice_loss',
                # 'bce_dice_loss',
                # 'tversky_loss',
                'log_cosh_dice_loss',
                'jacard_loss',
                'ssim_loss',
                'unet3p_hybrid_loss',
                'basnet_hybrid_loss',
            ]


def parse_args():
    """Parse arguments from terminal"""
    parser = argparse.ArgumentParser(description='Arguments of General training pipeline')

    parser.add_argument('--config', '-c', type=str, required=True       , help='Path of the config file')
    parser.add_argument('--device', '-d', type=int, default=0           , help='Device to run model')
    parser.add_argument('--output', '-o', type=str, default='./runs/LA/', help='Path of ouput folder')
    parser.add_argument('--loss'  , '-l', type=str, default='all'       , help='Type of loss')
    args = parser.parse_args()

    return args


def loss_function(y_true, y_pred, loss_type):
    """Ccmpute loss based on loss type"""
    loss_obj = SemanticLossFunctions()
    match loss_type:
        case 'weighted_cross_entropy_loss':
            loss = loss_obj.weighted_cross_entropy_loss(y_true, y_pred)
        case 'focal_loss':
            loss = loss_obj.focal_loss(y_true, y_pred)
        case 'dice_loss':
            loss = loss_obj.dice_loss(y_true, y_pred)
        case 'bce_dice_loss':
            loss = loss_obj.bce_dice_loss(y_true, y_pred)
        case 'tversky_loss':
            loss = loss_obj.tversky_loss(y_true, y_pred)
        case 'log_cosh_dice_loss':
            loss = loss_obj.log_cosh_dice_loss(y_true, y_pred)
        case 'jacard_loss':
            loss = loss_obj.jacard_loss(y_true, y_pred)
        case 'ssim_loss':
            loss = loss_obj.ssim_loss(y_true, y_pred)
        case 'unet3p_hybrid_loss':
            loss = loss_obj.unet3p_hybrid_loss(y_true, y_pred)
        case 'basnet_hybrid_loss':
            loss = loss_obj.basnet_hybrid_loss(y_true, y_pred)

    return loss


def load_model(model_backbone_type, feature_fuse, prediction='linear', mode='inference'):
    """Load model"""
    model = ModelBasedMit(
                            model_backbone_type=model_backbone_type,
                            feature_fuse=feature_fuse,
                            prediction=prediction,
                            mode=mode
                            )
    return model


def evaluate(config, model, dataset): # pylint: disable = W0621
    """Evaluate model"""    
    # --------------------------------------------------------------------
    # Switch model to eval mode and init output
    # --------------------------------------------------------------------
    model.eval()

    val = 0
    count = 0
    smooth = 1e-4

    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    val_path = os.path.join(config['dataset'], dataset, 'validation')
    val_images_path = os.path.join(val_path, 'images')
    val_masks_path = os.path.join(val_path, 'masks')
    val_loader = TestDataset(val_images_path,val_masks_path, init_trainsize)

    # --------------------------------------------------------------------
    # Validating pipleline
    # --------------------------------------------------------------------
    for _ in range(val_loader.size):
        image, gt, _ = val_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(DEVICE)

        pred, _ = model(image)
        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners = False)

        pred = pred.sigmoid()
        threshold = torch.tensor([0.5]).to(DEVICE)
        pred = (pred > threshold).float() * 1
        pred = pred.data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        target = np.array(gt)
        input_flat = np.reshape(pred,(-1))
        target_flat = np.reshape(target,(-1))

        intersection = input_flat*target_flat
        loss =  (2 * intersection.sum() + smooth) / (pred.sum() + target.sum() + smooth)

        a =  f'{loss:.4f}'
        a = float(a)

        val = val + a
        count = count + 1

    # --------------------------------------------------------------------
    # Switch back to train mode and return result
    # --------------------------------------------------------------------
    model.train()

    return val/count


def save_result(num_iter, config, model_path, dataset, output_path):
    """Save result (predicted image)"""
    save_path = os.path.join(output_path, str(num_iter).zfill(2), 'predict')
    os.makedirs(save_path, exist_ok=True)

    # --------------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------------
    model = torch.load(model_path).to(DEVICE)
    model.eval()

    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    test_path = os.path.join(config['dataset'], dataset, 'test')
    test_images_path = os.path.join(test_path, 'images')
    test_masks_path = os.path.join(test_path, 'masks')
    test_loader = TestDataset(test_images_path, test_masks_path, init_trainsize)


    # --------------------------------------------------------------------
    # Testing pipeline
    # --------------------------------------------------------------------
    for _ in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(DEVICE)

        pred = model(image)
        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
        pred = pred.sigmoid()
        threshold = torch.tensor([0.5]).to(DEVICE)
        pred = (pred > threshold).float() * 1
        pred = pred.data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        imageio.imwrite(os.path.join(save_path, name),img_as_ubyte(pred))


def plot_result(result_lst, save_path):
    """Plot loss after training"""
    epochs = range(1, len(result_lst) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, result_lst, label='Training Loss', marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)


def train_loop(config, num_iter, output_path, dataset, loss_type):
    """Train loop for 1 iteration"""
    # --------------------------------------------------------------------
    # Load from config
    # --------------------------------------------------------------------
    model_backbone_type = config['model']['model_backbone']
    feature_fuse = config['model']['feature_fuse']

    n_epochs = int(config['hyparameters']['n_epochs'])
    init_lr = float(config['hyparameters']['learning_rate'])
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    batch_size = int(config['hyparameters']['batch_size'])

    # --------------------------------------------------------------------
    # Clear GPU cache and load model
    # --------------------------------------------------------------------
    torch.cuda.empty_cache()
    model = load_model(model_backbone_type, feature_fuse, mode='train')
    model.to(DEVICE)
    lr = init_lr
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=50, gamma=0.3)

    # --------------------------------------------------------------------
    # Keep track of losses over time
    # --------------------------------------------------------------------
    losses = []
    coeff_max = 0

    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    train_path = os.path.join(config['dataset'], dataset, 'train')
    train_images_path = os.path.join(train_path, 'images')
    train_masks_path = os.path.join(train_path, 'masks')
    train_dataset = PolypDataset(train_images_path, train_masks_path, trainsize = init_trainsize, augmentations = True)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    iter_x = iter(train_loader)
    steps_per_epoch = len(iter_x)
    num_epoch = 0
    total_steps = (n_epochs+1)*steps_per_epoch

    # --------------------------------------------------------------------
    # Training pipleline
    # --------------------------------------------------------------------
    for step in range(1, total_steps):

        # ----------------------------------------------------------------
        # Reset iterators for each epoch
        # ----------------------------------------------------------------
        if step % steps_per_epoch == 0:
            iter_x = iter(train_loader)
            num_epoch = num_epoch + 1

        # ----------------------------------------------------------------
        # Compute loss in 1 step
        # ----------------------------------------------------------------
        images, masks = next(iter_x)
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        model.zero_grad()
        out, out_aux = model(images)
        out = F.interpolate(out, size=masks.shape[2:], mode='bilinear', align_corners=False)
        out_aux = F.interpolate(out_aux, size=masks.shape[2:], mode='bilinear', align_corners=False)
        out = out.sigmoid()
        out_aux = out_aux.sigmoid()

        loss = loss_function(masks, out, loss_type)
        loss_aux = loss_function(masks, out_aux, loss_type)
        total_loss = loss + loss_aux
        total_loss.backward()
        model_optimizer.step()

        # ----------------------------------------------------------------
        # Validate each epoch
        # ----------------------------------------------------------------
        # Print the log info
        if step % steps_per_epoch == 0:
            # ------------------------------------------------------------
            # Log loss of train
            losses.append(loss.item())
            msg = f'Epoch [{num_epoch:5d}/{n_epochs:5d}] | preliminary loss: {loss.item():6.6f} '
            aiLogger.info(msg)

            # ------------------------------------------------------------
            # Update lr
            scheduler.step()

            # ------------------------------------------------------------
            # Log loss of valid
            validation_coeff = evaluate(config, model, dataset)
            msg = f'Epoch [{num_epoch:5d}/{n_epochs:5d}] | validation coeffient: {validation_coeff:6.6f}'
            aiLogger.info(msg)

            # ------------------------------------------------------------
            # Save model if get better validation
            if coeff_max < validation_coeff:
                coeff_max = validation_coeff

                save_model_folder = os.path.join(output_path, str(num_iter).zfill(2))
                os.makedirs(save_model_folder, exist_ok=True)

                # Save all branch
                save_model_path = os.path.join(save_model_folder, 'model_all.pt')
                torch.save(model, save_model_path)

                # Save only inference brach
                save_model_path = os.path.join(save_model_folder, 'model.pt')
                torch.save(model.state_dict(), save_model_path)
                save_model = load_model(model_backbone_type, feature_fuse, mode='inference')
                save_model.load_state_dict(torch.load(save_model_path), strict=False)
                torch.save(save_model, save_model_path)

                msg = f'Save Optimized Model at Epoch [{num_epoch:5d}/{n_epochs:5d}]'
                aiLogger.info(msg)

    # --------------------------------------------------------------------
    # End training this iter - Save result
    # --------------------------------------------------------------------
    save_result(num_iter, config, save_model_path, dataset, output_path)

    loss_path = os.path.join(output_path, f'loss_{str(num_iter).zfill(2)}.png')
    plot_result(losses, loss_path)

    # --------------------------------------------------------------------
    # Load predict result to calculate all metrics
    # --------------------------------------------------------------------
    msg = 'Calculate all metrics for prediction on test dataset'
    aiLogger.info(msg)

    dice_val, iou_val, wfb_val, smeasure_val, emeasure_val = 0,0,0,0,0

    gt_path = os.path.join(config['dataset'], dataset, 'test')
    gt_path = os.path.join(gt_path, 'masks')
    pred_path = os.path.join(output_path, str(num_iter).zfill(2), 'predict')

    pred_img_path_lst = glob.glob(os.path.join(pred_path, '*'))

    for pred_img_path in tqdm(pred_img_path_lst):
        gt_img_path = os.path.join(gt_path, os.path.basename(pred_img_path))
        if not os.path.exists(gt_img_path):
            gt_img_path = gt_img_path[:-4] + '.jpg'

        result = calculate_image_pair(gt_img_path, pred_img_path)
        dice_val += result[0]
        iou_val += result[1]
        wfb_val += result[2]
        smeasure_val += result[3]
        emeasure_val += result[4]

    dice_val /=len(pred_img_path_lst)
    iou_val /=len(pred_img_path_lst)
    wfb_val /=len(pred_img_path_lst)
    smeasure_val /=len(pred_img_path_lst)
    emeasure_val /=len(pred_img_path_lst)

    with open(os.path.join(output_path, 'result.txt'), 'w', encoding='utf-8') as file:
        metric_result = f'Dice: {dice_val:.4f} - IoU: {iou_val:.4f} - WFb: {wfb_val:.4f} \
                        - sMeasure: {smeasure_val:.4f} - eMeasure: {emeasure_val:.4f}'
        file.write(metric_result)

    return losses, coeff_max


def train_repeats(config, output_dir, dataset, loss_type):
    """Train pipeline 1 repeat"""
    model_name = config['model']['model_name']
    repeats = int(config['hyparameters']['repeats'])

    # --------------------------------------------------------------------
    # Prepare folder
    # --------------------------------------------------------------------
    output_path = os.path.join(output_dir, model_name)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    # --------------------------------------------------------------------
    # Repeats pipline
    # --------------------------------------------------------------------
    for i in range(repeats):
        _, _ = train_loop(config, i+1, output_path, dataset, loss_type)


def main():
    """Main function"""
    global DEVICE # pylint: disable=W0601

    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{args.device}") # pylint: disable=W0621, C0103
    msg = f'Use DEVICE {DEVICE} for training'
    aiLogger.info(msg)

    if args.loss == 'all':
        aiLogger.info("Running all loss")
        for dataset in DATASET_LST:
            for loss in LOSS_LST:
                msg = 'Training '+config['model']['model_backbone']+f' with dataset {dataset}, loss {loss}'
                aiLogger.warning(msg)

                output_path = os.path.join(args.output, '_'.join([dataset, loss]))
                train_repeats(config, output_path, dataset, loss)
    else:
        for dataset in DATASET_LST:
            msg = 'Training '+config['model']['model_backbone']+f' with dataset {dataset}, loss {loss}'
            aiLogger.warning(msg)
            output_path = os.path.join(args.output, '_'.join([dataset, loss]))
            train_repeats(config, output_path, dataset, loss)


if __name__=="__main__":
    main()
