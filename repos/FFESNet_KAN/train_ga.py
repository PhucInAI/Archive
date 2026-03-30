"""Train Generalibity"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import imageio
from skimage import img_as_ubyte

import matplotlib.pyplot as plt

from FFESNet.model.model_based_mit import ModelBasedMit
from FFESNet.model.model_based_mamba import ModelBasedMamba
from FFESNet.utils.dataset import PolypDataset, TestDataset
from FFESNet.utils.losses.structure_loss import structure_loss
from FFESNet.utils.ai_logger import aiLogger

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_TEST = ['Kvasir', 'CVC-ColonDB', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-300']


def parse_args():
    """Parse arguments from terminal"""
    parser = argparse.ArgumentParser(description='Arguments of General training pipeline')

    parser.add_argument('--config', '-c', type=str, required=True       , help='Path of the config file')
    parser.add_argument('--output', '-o', type=str, default='./runs'    , help='Path of ouput folder')
    parser.add_argument('--type'  , '-t', type=str, default='PB'        , help='Type of training, PB or GA')
    args = parser.parse_args()

    return args


def load_model(model_backbone_type, feature_fuse, prediction='linear'):
    """Load model"""
    if model_backbone_type in ['B0', 'B1', 'B2', 'B3', 'B4']:
        model = ModelBasedMit(
                                model_backbone_type=model_backbone_type,
                                feature_fuse=feature_fuse,
                                prediction=prediction,
                             )
    elif model_backbone_type in ['T', 'T2', 'S', 'B', 'L', 'L2']:
        model = ModelBasedMamba(
                                model_backbone_type=model_backbone_type,
                                feature_fuse=feature_fuse,
                                prediction=prediction,
                               )

    return model


def evaluate(config, model): # pylint: disable = W0621
    """Evaluate model"""
    # --------------------------------------------------------------------
    # Switch model to eval mode and init output
    # --------------------------------------------------------------------
    model.eval()

    val = 0
    dataset_count = 0
    dataset_validation = []
    smooth = 1e-4

    # --------------------------------------------------------------------
    # Loop for each dataset
    # --------------------------------------------------------------------
    for data_name in DATASET_TEST:
        # ----------------------------------------------------------------
        # Set up data
        # ----------------------------------------------------------------
        init_trainsize = int(config['hyparameters']['init_trainsize'])
        val_path = os.path.join(config['dataset']['valid'], data_name)
        val_images_path = os.path.join(val_path, 'images')
        val_masks_path = os.path.join(val_path, 'masks')
        val_loader = TestDataset(val_images_path,val_masks_path, init_trainsize)

        # ----------------------------------------------------------------
        # Validating pipleline
        # ----------------------------------------------------------------

        count = 0
        total_mean_dice = 0

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

            dice = f'{loss:.4f}'
            dice = float(dice)
            total_mean_dice = total_mean_dice + dice
            count = count + 1

        dataset_validation.append(total_mean_dice/count)
        val = val + total_mean_dice/count
        dataset_count = dataset_count +1

    # --------------------------------------------------------------------
    # Switch back to train mode and return result
    # --------------------------------------------------------------------
    model.train()

    return val/dataset_count, dataset_validation


def save_result(num_iter, config, model_folder, output_path):
    """Save result (predicted image)"""
    for data_name in DATASET_TEST:
        save_path = os.path.join(output_path, str(num_iter).zfill(2), data_name)
        os.makedirs(save_path, exist_ok=True)

        # ----------------------------------------------------------------
        # Load model
        # ----------------------------------------------------------------
        model_path = os.path.join(model_folder, 'model_PB.pt')
        model = torch.load(model_path)
        model.eval()

        # ----------------------------------------------------------------
        # Set up data
        # ----------------------------------------------------------------
        init_trainsize = int(config['hyparameters']['init_trainsize'])
        test_path = os.path.join(config['dataset']['test'], data_name)
        test_images_path = os.path.join(test_path, 'images')
        test_masks_path = os.path.join(test_path, 'masks')
        test_loader = TestDataset(test_images_path, test_masks_path, init_trainsize)

        # ----------------------------------------------------------------
        # Testing pipeline
        # ----------------------------------------------------------------
        for _ in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(DEVICE)

            pred, _ = model(image)
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


def train_loop(config, num_iter, output_path):
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
    model = load_model(model_backbone_type, feature_fuse)
    if torch.cuda.device_count() > 1:
        msg = "Using", torch.cuda.device_count(), "GPUs!"
        aiLogger.info(msg)
    model = torch.nn.DataParallel(model)
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
    train_path = config['dataset']['train']
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

        loss = structure_loss(out, masks)
        loss_aux = structure_loss(out_aux, masks)
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
            validation_coeff, _ = evaluate(config, model)
            msg = f'Epoch [{num_epoch:5d}/{n_epochs:5d}] | validation coeffient: {validation_coeff:6.6f}'
            aiLogger.info(msg)

            # ------------------------------------------------------------
            # Save model if get better validation
            if coeff_max < validation_coeff:
                coeff_max = validation_coeff

                save_model_folder = os.path.join(output_path, str(num_iter).zfill(2))
                os.makedirs(save_model_folder, exist_ok=True)
                save_model_path = os.path.join(save_model_folder, 'model_PB.pt')
                torch.save(model, save_model_path)
                msg = f'Save Average Optimized Model at Epoch [{num_epoch:5d}/{n_epochs:5d}]'
                aiLogger.info(msg)

    # --------------------------------------------------------------------
    # End training this iter - Save result
    # --------------------------------------------------------------------
    save_result(num_iter, config, save_model_folder, output_path)

    loss_path = os.path.join(output_path, f'loss_{str(num_iter).zfill(2)}.png')
    plot_result(losses, loss_path)

    return losses, coeff_max


def train_repeats(config, output_dir):
    """Train pipeline 1 repeat"""
    model_name = config['model']['model_name']
    repeats = int(config['hyparameters']['repeats'])

    # --------------------------------------------------------------------
    # Prepare folder
    # --------------------------------------------------------------------
    output_path = os.path.join(output_dir, model_name)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # --------------------------------------------------------------------
    # Repeats pipline
    # --------------------------------------------------------------------
    for i in range(repeats):
        _, _ = train_loop(config, i+1, output_path)


def main():
    """Main function"""
    msg = f'Available DEVICE {DEVICE}'
    aiLogger.info(msg)

    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    train_repeats(config, args.output)


if __name__=="__main__":
    main()
