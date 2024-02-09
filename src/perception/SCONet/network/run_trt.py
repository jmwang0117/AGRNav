import os
import argparse
import torch
import torch.nn as nn
import sys
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from OCNet.common.seed import seed_all
from OCNet.common.config import CFG
from OCNet.common.dataset import get_dataset
from OCNet.common.io_tools import dict_to, _create_directory
from OCNet.common.logger import get_logger

def parse_args():
  parser = argparse.ArgumentParser(description='OCNet validating')
  parser.add_argument(
    '--weights',
    dest='weights_file',
    default='/home/melodic/Aerial-Walker/src/ocnet_ros/OCNet/weight/LMSCNet.pth',
    metavar='FILE',
    help='path to folder where model.pth file is',
    type=str,
  )
  parser.add_argument(
    '--dset_root',
    dest='dataset_root',
    default='/home/melodic/Aerial-Walker/src/oc_navigation/plan_manage/raw_data/voxels',
    metavar='DATASET',
    help='path to dataset root folder',
    type=str,
  )
  parser.add_argument(
    '--out_path',
    dest='output_path',
    default='/home/melodic/Aerial-Walker/src/ocnet_ros/OCNet/output',
    metavar='OUT_PATH',
    help='path to folder where predictions will be saved',
    type=str,
  )
  args = parser.parse_args()
  return args

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def run_trt_model(context, input_data, trt_engine):
    input_volume = np.prod(input_data.shape)
    inputs, outputs, bindings, stream = [], [], [], None
    for binding in trt_engine:
        size = trt.volume(trt_engine.get_binding_shape(binding)) * trt_engine.max_batch_size
        dtype = trt.nptype(trt_engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        bindings.append(int(device_mem))
        if trt_engine.binding_is_input(binding):
            inputs.append({"host_mem": host_mem, "device_mem": device_mem, "binding": binding})
        else:
            outputs.append({"host_mem": host_mem, "device_mem": device_mem, "binding": binding})

    stream = cuda.Stream()
    input_mem = inputs[0]["host_mem"].reshape(input_data.shape)
    input_mem = input_mem[:np.prod(input_data.shape)]  # 这将确保我们访问输入内存的正确大小
    
    input_mem = inputs[0]["host_mem"].reshape(np.prod(input_data.shape))
    np.copyto(input_mem, input_data.ravel())

    cuda.memcpy_htod_async(inputs[0]["device_mem"], inputs[0]["host_mem"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host_mem"], outputs[0]["device_mem"], stream)
    stream.synchronize()

    return outputs[0]["host_mem"].reshape(1, -1)


def test(trt_model_path, dset, _cfg, logger, out_path_root):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32 

    inv_remap_lut = dset.dataset.get_inv_remap_lut()

    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = load_engine(trt_runtime, trt_model_path)
    context = engine.create_execution_context()

    with torch.no_grad():
        for t, (data, indices) in enumerate(dset):

            data = dict_to(data, device, dtype)
            input_data = data['3D_OCCUPANCY'].cpu().numpy()

            start_time = time.time()
            result = run_trt_model(context, input_data, engine)
            print("Result shape:", result.shape)
            end_time = time.time()

            inference_time = end_time - start_time
            fps = 1 / inference_time
            logger.info("Inference time: {:.4f} seconds, FPS: {:.2f}".format(inference_time, fps))

            scores = np.argmax(result, axis=1)
            
            curr_index = 0
            for score in scores:
                print(score.shape)
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
  dataset = get_dataset(_cfg)['val']

  logger.info('=> Loading network architecture...')
  trt_model_path = "/home/melodic/Aerial-Walker/src/ocnet_ros/OCNet/weight/LMSCNet.trt"
  


  test(trt_model_path, dataset, _cfg, logger, out_path_root)
  logger.info('=> ============ Network Test Done ============')

  exit()
  
if __name__ == '__main__':
    main()