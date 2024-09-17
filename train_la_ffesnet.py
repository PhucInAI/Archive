"""Train FFESNet"""


import os
import shutil
import argparse
import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import imageio
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

from FFESNet.model.model_based_mit import ModelBasedMit
from FFESNet.utils.dataset import PolypDataset, TestDataset
from FFESNet.utils.ai_logger import aiLogger


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Arguments of Learning Ability training pipeline')

    parser.add_argument('--config', '-c', type=str, required=True       , help='Path of the config file')
    parser.add_argument('--output', '-o', type=str, default='./runs'    , help='Path of ouput folder')
    parser.add_argument('--name'  , '-n', type=str, default=''          , help='Name of output model')

    args = parser.parse_args()

    return args


def load_model(model_backbone_type, feature_fuse, prediction='linear'):
    """Load model"""
    model = ModelBasedMit(
                            model_backbone_type=model_backbone_type,
                            feature_fuse=feature_fuse,
                            prediction=prediction
                         )

    return model


def evaluate(config, model):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ESFPNet.eval()

    global device

    val = 0
    count = 0
    smooth = 1e-4

    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    val_path = config['dataset']['valid']
    val_images_path = os.path.join(val_path, 'images')
    val_masks_path = os.path.join(val_path, 'masks')
    val_loader = TestDataset(val_images_path,val_masks_path, init_trainsize)

    # --------------------------------------------------------------------
    # Validating pipleline
    # --------------------------------------------------------------------
    model.eval()

    for i in range(val_loader.size):
        image, gt, name = val_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.to(device)

        pred, pred_aux = model(image)

        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners = False)
        pred = pred.sigmoid()
        threshold = torch.tensor([0.5]).to(device)
        pred = (pred > threshold).float() * 1
        pred = pred.data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target = np.array(gt)
        input_flat = np.reshape(pred,(-1))
        target_flat = np.reshape(target,(-1))
        intersection = (input_flat*target_flat)
        loss =  (2 * intersection.sum() + smooth) / (pred.sum() + target.sum() + smooth)

        # pred_aux = F.upsample(pred_aux, size=gt.shape, mode='bilinear', align_corners = False)
        # pred_aux = pred_aux.sigmoid()
        # threshold = torch.tensor([0.5]).to(device)
        # pred_aux = (pred_aux > threshold).float() * 1
        # pred_aux = pred_aux.data.cpu().numpy().squeeze()
        # pred_aux = (pred_aux - pred_aux.min()) / (pred_aux.max() - pred_aux.min() + 1e-8)
        # target = np.array(gt)
        # input_flat = np.reshape(pred_aux,(-1))
        # target_flat = np.reshape(target,(-1))
        # intersection = (input_flat*target_flat)
        # loss_aux =  (2 * intersection.sum() + smooth) / (pred_aux.sum() + target.sum() + smooth)

        # loss = loss + 0.5*loss_aux

        a =  '{:.4f}'.format(loss)
        a = float(a)

        val = val + a
        count = count + 1

    model.train()

    return val/count


def save_result(numIters, config, model_path):
    global device
    global output_path

    save_path = os.path.join(output_path, str(numIters).zfill(2), 'predict')
    os.makedirs(save_path, exist_ok=True)

    # --------------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------------
    model = torch.load(model_path)
    model.eval()

    # --------------------------------------------------------------------
    # Set up data
    # --------------------------------------------------------------------
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    test_path = config['dataset']['test']
    test_images_path = os.path.join(test_path, 'images')
    test_masks_path = os.path.join(test_path, 'masks')
    test_loader = TestDataset(test_images_path, test_masks_path, init_trainsize)


    # --------------------------------------------------------------------
    # Testing pipeline
    # --------------------------------------------------------------------
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)

        pred, _ = model(image)
        pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
        pred = pred.sigmoid()
        threshold = torch.tensor([0.5]).to(device)
        pred = (pred > threshold).float() * 1
        pred = pred.data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        imageio.imwrite(os.path.join(save_path, name),img_as_ubyte(pred))


def plot_result(result_lst, save_path):
    epochs = range(1, len(result_lst) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, result_lst, label='Training Loss', marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)


def train_loop(config, numIters):
    global device
    global output_path

    model_type = config['model']['model_backbone']

    n_epochs = int(config['hyparameters']['n_epochs'])
    init_lr = float(config['hyparameters']['learning_rate'])
    init_trainsize = int(config['hyparameters']['init_trainsize'])
    batch_size = int(config['hyparameters']['batch_size'])

    # --------------------------------------------------------------------
    # Clear GPU cache and load model
    # --------------------------------------------------------------------
    torch.cuda.empty_cache()
    model = load_model(model_type)
    model.to(device)
    lr = init_lr

    sigma1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    sigma2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    model_optimizer = torch.optim.AdamW([
    {'params': list(model.parameters())},
    {'params': [sigma1, sigma2]}
], lr=lr)


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
    trainDataset = PolypDataset(train_images_path, train_masks_path, trainsize = init_trainsize, augmentations = True)
    train_loader = DataLoader(dataset = trainDataset, batch_size = batch_size, shuffle = True)

    iter_X = iter(train_loader)
    steps_per_epoch = len(iter_X)
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
            iter_X = iter(train_loader)
            num_epoch = num_epoch + 1

        # ----------------------------------------------------------------
        # Compute loss in 1 step
        # ----------------------------------------------------------------
        images, masks = next(iter_X)
        images = images.to(device)
        masks = masks.to(device)

        model.zero_grad()
        out, out_aux = model(images)
        out = F.interpolate(out, size=masks.shape[2:], mode='bilinear', align_corners=False)
        out_aux = F.interpolate(out_aux, size=masks.shape[2:], mode='bilinear', align_corners=False)

        if 'loss' in config['hyparameters']:
            loss_name = config['hyparameters']['loss']
        else:
            loss_name = 'structure'
        loss_function = load_loss(loss_name)

        loss = loss_function(out, masks)
        loss_aux = loss_function(out_aux, masks)
        total_loss = (1.0 / (2 * sigma1 ** 2)) * loss + \
                     (1.0 / (2 * sigma2 ** 2)) * loss_aux + \
                     torch.log(sigma1 * sigma2)

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
            aiLogger.info('Epoch [{:5d}/{:5d}] | preliminary loss: {:6.6f} '.format(num_epoch, n_epochs, loss.item()))
            aiLogger.info(sigma1)
            aiLogger.info(sigma2)

            # ------------------------------------------------------------
            # Log loss of valid
            validation_coeff = evaluate(config, model)
            aiLogger.info('Epoch [{:5d}/{:5d}] | validation coeffient: {:6.6f} '.format(num_epoch, n_epochs, validation_coeff))

            # ------------------------------------------------------------
            # Save model if get better validation
            if coeff_max < validation_coeff:
                coeff_max = validation_coeff

                save_model_folder = os.path.join(output_path, str(numIters).zfill(2))
                os.makedirs(save_model_folder, exist_ok=True)
                save_model_path = os.path.join(save_model_folder, 'model_{}_{}.pt'.format(numIters, num_epoch))
                torch.save(model, save_model_path)
                aiLogger.info('Save Learning Ability Optimized Model at Epoch [{:5d}/{:5d}]'.format(num_epoch, n_epochs))


    save_result(numIters, config, save_model_path)
    loss_path = os.path.join(output_path, 'loss_{}.png'.format(str(numIters)).zfill(2))
    plot_result(losses, loss_path)
    return losses, coeff_max


def train_repeats(config, output_dir, model_name):
    """Train pipeline 1 repeat"""
    global device
    global output_path

    if model_name == '':
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
        losses, coeff_max = train_loop(config, i+1)


def main():
    args = parse_args()

    global device
    global config

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    train_repeats(config, args.output, args.name)


if __name__=="__main__":
    main()
