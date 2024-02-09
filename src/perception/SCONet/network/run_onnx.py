import os
import argparse
import torch
import torch.nn as nn
import sys
import numpy as np
import time
import onnxruntime as ort
# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from LMSCNet.common.seed import seed_all
from LMSCNet.common.config import CFG
from LMSCNet.common.dataset import get_dataset
from LMSCNet.common.model import get_model
from LMSCNet.common.logger import get_logger
from LMSCNet.common.io_tools import dict_to, _create_directory
import LMSCNet.common.checkpoint as checkpoint


def parse_args():
  parser = argparse.ArgumentParser(description='LMSCNet validating')
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

def run_onnx_model(onnx_session, input_data):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_data})
    return result[0]
  
def test(onnx_session, dset, _cfg, logger, out_path_root):
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  dtype = torch.float32  # Tensor type to be used
  inv_remap_lut = dset.dataset.get_inv_remap_lut()

  with torch.no_grad():

    for t, (data, indices) in enumerate(dset):

      data = dict_to(data, device, dtype)
      
      # Apply ONNX inference instead of PyTorch model
      input_data = data['3D_OCCUPANCY'].cpu().numpy()
      start_time = time.time()  # Record time before inference
      result = run_onnx_model(onnx_session, input_data)
      end_time = time.time()  # Record time after inference

      inference_time = end_time - start_time  # Calculate inference time in seconds
      fps = 1 / inference_time  # Calculate FPS (frames per second)

      # Print or log FPS
      logger.info("Inference time: {:.4f} seconds, FPS: {:.2f}".format(inference_time, fps))
      
      scores = {'pred_semantic_1_1': result}
      for key in scores:
        scores[key] = np.argmax(scores[key], axis=1)

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

  return

def main():
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
  onnx_model_path = "/root/LMSCNet/weight/LMSCNet.onnx"
  # Load ONNX model
  # Load the ONNX model with GPU execution
  providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else None
  onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)


  test(onnx_session, dataset, _cfg, logger, out_path_root)

  logger.info('=> ============ Network Test Done ============')

  exit()


if __name__ == '__main__':
  main()