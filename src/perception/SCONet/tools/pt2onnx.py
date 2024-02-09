import torch
import os
import argparse
import torch
import torch.nn as nn
import sys
import logging
# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from network.common.seed import seed_all
from network.common.config import CFG
from network.common.dataset import get_dataset
from network.common.model import get_model
from network.common.logger import get_logger
from network.common.io_tools import dict_to
from network.common.metrics import Metrics
import network.common.checkpoint as checkpoint

torch.backends.cudnn.enabled = False

weights_f = '/root/LMSCNet/weight/LMSCNet.pth'  # Change this to your pretrained model path
dataset_f = '/root/datasets/semantic_kitti'  # Change this to your dataset root path

assert os.path.isfile(weights_f), '=> No file found at {}'

checkpoint_path = torch.load(weights_f)
config_dict = checkpoint_path.pop('config_dict')
config_dict['DATASET']['ROOT_DIR'] = dataset_f

# Read train configuration file
_cfg = CFG()
_cfg.from_dict(config_dict)
# Setting the logger to print statements and also save them into logs file
logger = get_logger(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'logs_val.log')

logger.info('============ Validation weights: "%s" ============\n' % weights_f)

def get_input_dimensions(train_dataset):
    temp_data, _ = train_dataset.__getitem__(0)
    input_data = temp_data["3D_OCCUPANCY"]
    input_dimensions = input_data.shape[1:]
    return input_dimensions


dataset = get_dataset(_cfg)
# Determine input_dimensions based on the dataset
input_dimensions = get_input_dimensions(dataset['train'].dataset)



# Load model
model = get_model(_cfg, dataset['train'].dataset)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model = model.module


model = checkpoint.load_model(model, weights_f, logger)

nbr_iterations = len(dataset['val'])
metrics = Metrics(dataset['val'].dataset.nbr_classes, nbr_iterations, model.get_scales())
metrics.reset_evaluator()
metrics.losses_track.set_validation_losses(model.get_validation_loss_keys())
metrics.losses_track.set_train_losses(model.get_train_loss_keys())

ONNX_MODEL_PATH = '/root/LMSCNet/weight/LMSCNet.onnx'  # Change this to where you want to save ONNX model

# Determine the input shape from the config

dummy_input = torch.randn(1, 1, *input_dimensions)  # Assuming your model expects 1 channel image
dummy_input = {'3D_OCCUPANCY': dummy_input}  # as your forward method needs a dictionary as input

# Export the model
torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH)