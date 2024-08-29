import os
import glob
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
import warnings
warnings.filterwarnings("ignore")
import logging
import matplotlib.pyplot as plt


from FFESNet.models.model import Model
from FFESNet.utils.dataset import PolypDataset, TestDataset
from FFESNet.utils.losses.structure_loss import structure_loss
from FFESNet.utils.ai_logger import aiLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments of Learning Ability training pipeline')

    parser.add_argument('--config', '-c', type=str, required=True       , help='Path of the config file')
    parser.add_argument('--device', '-d', type=int, required=True       , help='Device to use')
    parser.add_argument('--output', '-o', type=str, default='./runs'    , help='Path of ouput folder')
    parser.add_argument('--type'  , '-t', type=str, default='PB'        , help='Type of training, PB or GA')
    args = parser.parse_args()

    return args


def load_model(model_type):
    """Load model"""
    model = Model(model_type=model_type)

    return model


def evaluate(config, model):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ESFPNet.eval()

    global device

    model.eval()

    val = 0
    dataset_count = 0
    datasetValidation = []
    smooth = 1e-4

    for data_name in datasetTest:
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
        total_meanDice = 0

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

            Dice =  '{:.4f}'.format(loss)
            Dice = float(Dice)
            total_meanDice = total_meanDice + Dice
            count = count + 1

        datasetValidation.append(total_meanDice/count)
        val = val + total_meanDice/count
        dataset_count = dataset_count +1

    model.train()

    return val/dataset_count, datasetValidation


def save_result(numIters, config, model_folder):
    global device
    global output_path

    # --------------------------------------------------------------------
    # Predict PB
    # --------------------------------------------------------------------
    for data_name in datasetTest:

        save_path = os.path.join(output_path, str(numIters).zfill(2), 'predict_PB', data_name)
        os.makedirs(save_path, exist_ok=True)

        # ----------------------------------------------------------------
        # Load model
        # ----------------------------------------------------------------
        model_path = os.path.join(model_folder, 'model_PB.pt'.format(data_name))
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
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)

            pred, pred_aux = model(image)
            pred = F.upsample(pred, size=gt.shape, mode='bilinear', align_corners=False)
            pred = pred.sigmoid()
            threshold = torch.tensor([0.5]).to(device)
            pred = (pred > threshold).float() * 1
            pred = pred.data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            imageio.imwrite(os.path.join(save_path, name),img_as_ubyte(pred))


    # --------------------------------------------------------------------
    # Predict GA
    # --------------------------------------------------------------------
    for data_name in datasetTest:

        save_path = os.path.join(output_path, str(numIters).zfill(2), 'predict_GA', data_name)
        os.makedirs(save_path, exist_ok=True)

        # ----------------------------------------------------------------
        # Load model
        # ----------------------------------------------------------------
        model_path = os.path.join(model_folder, 'model_GA_{}.pt'.format(data_name))
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
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)

            pred, pred_aux = model(image)
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
    # Setup log
    # --------------------------------------------------------------------
    # # Set up the logging configuration
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # # Create a file handler that writes log messages to a file
    # file_handler = logging.FileHandler(os.path.join(output_path, 'logfile_{}.log'.format(str(numIters).zfill(2))))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # # Create a stream handler that writes log messages to the console
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    # # Get the root logger and add the handlers
    # logger = logging.getLogger()
    # aiLogger.addHandler(file_handler)
    # aiLogger.addHandler(stream_handler)

    # --------------------------------------------------------------------
    # Clear GPU cache and load model
    # --------------------------------------------------------------------
    torch.cuda.empty_cache()
    model = load_model(model_type)
    model.to(device)
    lr = init_lr
    # sigma1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    # sigma2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    model_optimizer = torch.optim.AdamW([
        {'params': list(model.parameters())},
        # {'params': [sigma1, sigma2]}
    ], lr=lr)

    # model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=25, gamma=0.25)

    # --------------------------------------------------------------------
    # Keep track of losses over time
    # --------------------------------------------------------------------
    losses = []
    coeff_max = 0
    datasetValidation_max = [0,0,0,0,0]

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

        loss = structure_loss(out, masks)
        loss_aux = structure_loss(out_aux, masks)
        # total_loss = (1.0 / (2 * sigma1 ** 2)) * loss + \
        #              (1.0 / (2 * sigma2 ** 2)) * loss_aux + \
        #              torch.log(sigma1 * sigma2)
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
            aiLogger.info('Epoch [{:5d}/{:5d}] | preliminary loss: {:6.6f} '.format(num_epoch, n_epochs, loss.item()))

            # ------------------------------------------------------------
            # Update lr
            scheduler.step()

            # ------------------------------------------------------------
            # Log loss of valid
            validation_coeff, datasetValidation = evaluate(config, model)
            aiLogger.info('Epoch [{:5d}/{:5d}] | validation coeffient: {:6.6f} '.format(num_epoch, n_epochs, validation_coeff))

            # ------------------------------------------------------------
            # Save model PB if get better validation
            if coeff_max < validation_coeff:
                coeff_max = validation_coeff

                save_model_folder = os.path.join(output_path, str(numIters).zfill(2))
                os.makedirs(save_model_folder, exist_ok=True)
                save_model_path = os.path.join(save_model_folder, 'model_PB.pt')
                torch.save(model, save_model_path)
                aiLogger.info('Save Average Optimized Model at Epoch [{:5d}/{:5d}]'.format(num_epoch, n_epochs))


            # ------------------------------------------------------------
            # Save model GA if get better datasetValidation
            for dataset_index, dataset_name in enumerate(datasetTest):
                if datasetValidation_max[dataset_index] < datasetValidation[dataset_index]:
                    datasetValidation_max[dataset_index] = datasetValidation[dataset_index]

                    save_model_folder = os.path.join(output_path, str(numIters).zfill(2))
                    os.makedirs(save_model_folder, exist_ok=True)
                    save_model_path = os.path.join(save_model_folder, 'model_GA_{}.pt'.format(dataset_name))
                    torch.save(model, save_model_path)
                    aiLogger.info('Save Optimized {} Model at Epoch [{:5d}/{:5d}, with coefficient: {:6.6f}]'
                                .format(dataset_name, num_epoch, n_epochs, datasetValidation[dataset_index]))



    save_result(numIters, config, save_model_folder)
    loss_path = os.path.join(output_path, 'loss_{}.png'.format(str(numIters)).zfill(2))
    plot_result(losses, loss_path)
    return losses, coeff_max


def train_repeats(config, output_dir):
    """Train pipeline 1 repeat"""
    global device
    global output_path

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
    global datasetTest

    if args.type == 'PB':
        datasetTest = ['Kvasir', 'CVC-ColonDB', 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'CVC-300']
    else: # GA
        datasetTest = ['CVC-ColonDB', 'ETIS-LaribPolypDB']


    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    train_repeats(config, args.output)


if __name__=="__main__":
    main()