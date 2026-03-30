"""
Unlearn configuration for FedRIR.

This configuration file closely mirrors the original settings from the
Ferrari implementation, with the addition of a new argument to control
the number of convolutional filters removed per layer during the
unlearning step.  Removing multiple filters can offer finer control
over how aggressively features are excised from the model.
"""

import argparse

# Instantiate the argument parser.  See the accompanying README for a
# description of the supported options.
parser = argparse.ArgumentParser()

# Device configuration
parser.add_argument("-gpu", action="store_true", help="use gpu or not")

# Model configuration
parser.add_argument("-weight_path", type=str, default=None,
                    help="Path to pretrained model weights")
parser.add_argument("-hidden_layer_num", type=int,
                    help="hidden layer size for tabular models")
parser.add_argument("-save_model", action='store_true',
                    help="save the unlearned model to disk")
parser.add_argument("-checkpoint", type= str, help= "model folder path", default='/home/ptn/Storage/Research/2025/FedRIR/assets/models')

# Dataset configuration
parser.add_argument("-root", type= str, help= "data folder path", default="/home/ptn/Storage/Research/2025/FedRIR/assets")
parser.add_argument("-preprocessed_dir", type=str, default="preprocessed_data")
parser.add_argument("-dataset", type=str, default="Cifar10",
                    choices=["MNist", "FMNist", "Cifar10", "Cifar20",
                             "Cifar100", "Celeba", "adult", "diabetes"],
                    help="dataset for feature unlearning")
parser.add_argument("-save_preprocessed", action="store_true",
                    help="option to save the preprocessed data into pkl")

# Federated learning client configuration
parser.add_argument('-client_num', type=int, help="number of clients: K")
parser.add_argument('-frac', type=float, default=0.4,
                    help='fraction of clients per round: C')
parser.add_argument('-unlearn_client_index', type=int, default=0,
                    help="index of the client to unlearn")

# Sensitive feature parameters
parser.add_argument("-reduced", action='store_true',
                    help="reduce the number of samples for debugging")
parser.add_argument("-celeba_classification", type=int,
                    help="classification task for Celeba (e.g. gender)")
parser.add_argument("-celeba_bias_feature", type=int,
                    help="feature index used for bias training")
parser.add_argument("-pertubbed_part", type=str,
                    choices=["mouth", "eye", "nose", "face", "face_except_mouth"],
                    help="facial region to perturb when generating sensitive data")

# Backdoor parameters
parser.add_argument("-trigger_size", type=int,
                    help="square size of the backdoor trigger")
parser.add_argument("-trigger_label", type=int,
                    help="label associated with the backdoor trigger")

# Bias parameters
parser.add_argument("-mnist_mode", type=str, choices=["digit", "background"],
                    help="bias mode for MNIST")
parser.add_argument("-bias_ratio", type=float,
                    help="ratio of biased samples in the dataset")

# Unlearning hyperparameters
parser.add_argument("-n_epochs", type=int, default=1,
                    help="number of epochs for the unlearning method")
parser.add_argument("-lambda_coef", type=float, default=1.0,
                    help="coefficient controlling the Lipschitz term (unused in FedRIR)")
parser.add_argument("-sigma", type=float, help="Gaussian noise standard deviation")
parser.add_argument("-sample_number", type=int,
                    help="number of perturbed samples per original image")
parser.add_argument("-min_sigma", type=float, help="minimum sigma for noise sampling")
parser.add_argument("-max_sigma", type=float, help="maximum sigma for noise sampling")
parser.add_argument("-batch_size", type=int, help="mini-batch size for the dataloaders")
parser.add_argument("-lr", type=float, help="learning rate for potential optimisation steps")

# Unlearning configuration
parser.add_argument("-unlearn", action="store_false",
                    help="if set, skip unlearning and evaluate the baseline model")
parser.add_argument("-unlearning_scenario", type=str,
                    choices=["sensitive", "backdoor", "bias"],
                    help="scenario specifying which type of feature to unlearn")
parser.add_argument("-lipschitz_mode", type=str, default="multiple",
                    choices=["single", "multiple"],
                    help="noise sampling mode (retained for backwards compatibility)")

# New FedRIR parameter
parser.add_argument("-n_filters", type=int, default=1,
                    help="number of convolutional filters to remove per layer during FedRIR")

# Random seed
parser.add_argument("-seed", type=int, help="random seed for reproducibility")

# Parse arguments when this module is imported
arguments = parser.parse_args()