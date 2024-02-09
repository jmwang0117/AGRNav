#!/usr/bin/env python3
# -*-coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import sys
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
from network.common.seed import seed_all
from network.common.config import CFG
from network.common.dataset import get_dataset
from network.common.logger import get_logger
from network.common.io_tools import dict_to, _create_directory
import network.data.io_data as SemanticKittiIO


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
    input_mem = input_mem[:np.prod(input_data.shape)]

    input_mem = inputs[0]["host_mem"].reshape(np.prod(input_data.shape))
    np.copyto(input_mem, input_data.ravel())

    cuda.memcpy_htod_async(inputs[0]["device_mem"], inputs[0]["host_mem"], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]["host_mem"], outputs[0]["device_mem"], stream)
    stream.synchronize()

    return outputs[0]["host_mem"].reshape(1, -1)


def publish_coordinates(coordinates, publisher):
    coordinates = coordinates[:, [0, 2, 1]]
    coordinates_msg = Float64MultiArray()

    for coordinate in coordinates:
        print(f"coordinate : {coordinate}")
        coordinates_msg.data.extend(coordinate)

    publisher.publish(coordinates_msg)


def test(trt_model_path, dset, _cfg, logger, out_path_root, coordinates_publisher):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32

    inv_remap_lut = dset.dataset.get_inv_remap_lut()
    inference_time = []
    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = load_engine(trt_runtime, trt_model_path)
    context = engine.create_execution_context()

    with torch.no_grad():
        for t, (data, indices) in enumerate(dset):

            data = dict_to(data, device, dtype)
            input_data = data['3D_OCCUPANCY'].cpu().numpy()
            # Record the inference start time
            inference_start_time = time.time()

            result = run_trt_model(context, input_data, engine)
            print("Result shape:", result.shape)
            # Record the inference end time
            inference_end_time = time.time()
            
            # Log the inference time of each sample
            inference_time.append(inference_end_time - inference_start_time)

            scores = np.argmax(result, axis=1)
            for key in scores:
                scores[key] = torch.argmax(scores[key], dim=1).data.cpu().numpy()
            curr_index = 0
            for score in scores['pred_semantic_1_1']:
                 # voxel occupancy file
                input_filename = dset.dataset.filepaths['3D_OCCUPANCY'][indices[curr_index]]
                print(input_filename)

                # Read the voxel occupancy from the file
                voxel_occupancy = SemanticKittiIO._read_occupancy_SemKITTI(input_filename)

                # Reshape the voxel occupancy array to the correct dimensions
                voxel_occupancy = voxel_occupancy.reshape(256, 32, 256)

                # Create a mask for occupied voxels
                voxel_mask = voxel_occupancy.ravel() == 1

                # Count the occupied voxels in the voxel file
                voxel_occupied_count = np.count_nonzero(voxel_mask)

                # Create a mask for occupied voxels in scores
                score_mask = score.ravel() > 0

                # Count the occupied voxels in scores
                score_occupied_count = np.count_nonzero(score_mask)

                # Compute the intersection of occupied voxels in both score and voxel_occupancy
                intersection = np.logical_and(voxel_mask, score_mask)

                # Count the intersected occupied voxels
                intersection_count = np.count_nonzero(intersection)

                # Compute the non-intersected occupied voxels coordinates in voxel_occupancy
                non_intersection = np.logical_and(score_mask, np.logical_not(voxel_mask))

                # Get the non-intersected occupied voxel coordinates
                non_intersection_coordinates = np.column_stack(np.nonzero(non_intersection.reshape(256, 32, 256)))

                publish_coordinates(non_intersection_coordinates, coordinates_publisher)
                
                score = np.moveaxis(score, [0, 1, 2], [0, 2, 1]).reshape(-1).astype(np.uint16)
                score = inv_remap_lut[score].astype(np.uint16)
                
              
                filename, extension = os.path.splitext(os.path.basename(input_filename))
                out_filename = os.path.join(out_path_root, 'predictions', filename + '.label')
                _create_directory(os.path.dirname(out_filename))
                score.tofile(out_filename)
                # shutil.copy(input_filename, ori_voxels_path)
                os.remove(input_filename)
                curr_index += 1
                
    return inference_time


def main():
    rospy.init_node("inference_node")
    #Create the publisher using a specific ROS message type and topic
    coordinates_publisher = rospy.Publisher('/non_intersection_coordinates', Float64MultiArray, queue_size=1000)

    
    torch.backends.cudnn.enabled = False
    seed_all(0)
    weights_f = rospy.get_param('~weights_file')
    dataset_f = rospy.get_param('~dataset_root')
    out_path_root = rospy.get_param('~output_path')

    assert os.path.isfile(weights_f), '=> No file found at {}'

    checkpoint_path = torch.load(weights_f)
    config_dict = checkpoint_path.pop('config_dict')
    config_dict['DATASET']['ROOT_DIR'] = dataset_f

    _cfg = CFG()
    _cfg.from_dict(config_dict)
    logger = get_logger(out_path_root, 'logs_test.log')
    logger.info('============ Test weights: "%s" ============\n' % weights_f)

    wait_time = 2  # Seconds to wait before checking the dataset folder again
    train_batch_size = 6  # Set your desired batch size here
    
    while not rospy.is_shutdown():
        
        dataset = None
        while dataset is None:
            # Check if the dataset folder has sufficient data (files) for the batch size
            dataset_files = os.listdir(dataset_f)
            if len(dataset_files) >= train_batch_size:
                dataset = get_dataset(_cfg)['test']
            else:
                rospy.loginfo("Waiting for dataset folder to accumulate sufficient files.")
                rospy.sleep(wait_time)
                
        logger.info('=> Loading network architecture...')
        trt_model_path = "/home/melodic/Aerial-Walker/src/ocnet_ros/OCNet/weight/LMSCNet.trt"
        rate = rospy.Rate(10)
        inference_time = test(trt_model_path, dataset, _cfg, logger, out_path_root, coordinates_publisher)
        logger.info('Inference time per frame is %.4f seconds\n' % (np.sum(inference_time) / 6.0))
        logger.info('=> ============ Network Test Done ============')
        rate.sleep()

  
if __name__ == '__main__':
    main()