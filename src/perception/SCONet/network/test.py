import os
import argparse
import torch
import torch.nn as nn
import sys
import numpy as np

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from network.common.seed import seed_all
from network.common.config import CFG
from network.common.dataset import get_dataset
from network.common.model import get_model
from network.common.logger import get_logger
from network.common.io_tools import dict_to, _create_directory
import network.common.checkpoint as checkpoint


def parse_args():
  parser = argparse.ArgumentParser(description='SCONet validating')
  parser.add_argument(
    '--weights',
    dest='weights_file',
    default='',
    metavar='FILE',
    help='path to folder where model.pth file is',
    type=str,
  )
  parser.add_argument(
    '--dset_root',
    dest='dataset_root',
    default='',
    metavar='DATASET',
    help='path to dataset root folder',
    type=str,
  )
  parser.add_argument(
    '--out_path',
    dest='output_path',
    default='',
    metavar='OUT_PATH',
    help='path to folder where predictions will be saved',
    type=str,
  )
  args = parser.parse_args()
  return args


import time

def test(model, dset, _cfg, logger, out_path_root):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32  # Tensor type to be used

    # Moving optimizer and model to used device
    model = model.to(device=device)
    logger.info('=> Passing the network on the test set...')
    model.eval()
    inv_remap_lut = dset.dataset.get_inv_remap_lut()

    start_time = time.time()
    inference_time = 0

    with torch.no_grad():
        for t, (data, indices) in enumerate(dset):
            data = dict_to(data, device, dtype)

            inference_start_time = time.time()
            scores = model(data)
            inference_end_time = time.time()
            inference_time += (inference_end_time - inference_start_time)

            for key in scores:
                scores[key] = torch.argmax(scores[key], dim=1).data.cpu().numpy()

            curr_index = 0
            for score in scores['pred_semantic_1_1']:
                score = np.moveaxis(score, [0, 1, 2], [0, 2, 1]).reshape(-1).astype(np.uint16)
                score = inv_remap_lut[score].astype(np.uint16)
                input_filename = dset.dataset.filepaths['3D_OCCUPANCY'][indices[curr_index]]
                filename, extension = os.path.splitext(os.path.basename(input_filename))
                sequence = os.path.dirname(input_filename).split('/')[-2]
                out_filename = os.path.join(out_path_root, 'sequences', sequence, 'predictions', filename + '.label')
                _create_directory(os.path.dirname(out_filename))
                score.tofile(out_filename)
                logger.info('=> Sequence {} - File {} saved'.format(sequence, os.path.basename(out_filename)))
                curr_index += 1

    elapsed_time = time.time() - start_time
    fps = len(dset) / inference_time
    logger.info('Inference FPS: {:.2f}'.format(fps))
    return


def main():

  # https://github.com/pytorch/pytorch/issues/27588
  torch.backends.cudnn.enabled = False

  seed_all(0)

  args = parse_args()

  weights_f = args.weights_file
  dataset_f = args.dataset_root
  out_path_root = args.output_path

  assert os.path.isfile(weights_f), '=> No file found at {}'

  checkpoint_path = torch.load(weights_f)
  config_dict = checkpoint_path.pop('config_dict')
  config_dict['DATASET']['ROOT_DIR'] = dataset_f

  # Read train configuration file
  _cfg = CFG()
  _cfg.from_dict(config_dict)
  # Setting the logger to print statements and also save them into logs file
  logger = get_logger(out_path_root, 'logs_test.log')

  logger.info('============ Test weights: "%s" ============\n' % weights_f)
  dataset = get_dataset(_cfg)['test']

  logger.info('=> Loading network architecture...')
  model = get_model(_cfg, dataset.dataset)
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model = model.module

  logger.info('=> Loading network weights...')
  model = checkpoint.load_model(model, weights_f, logger)

  test(model, dataset, _cfg, logger, out_path_root)

  logger.info('=> ============ Network Test Done ============')

  exit()


if __name__ == '__main__':
  main()