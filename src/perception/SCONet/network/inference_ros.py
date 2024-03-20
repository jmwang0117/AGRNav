#!/usr/bin/env python3
# -*-coding:utf-8 -*-
import shutil
import os
import torch
import torch.nn as nn
import sys
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import time
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
import network.data.io_data as SemanticKittiIO



def publish_coordinates(coordinates, publisher):
    
    coordinates = coordinates[:, [0, 2, 1]]
    coordinates_msg = Float64MultiArray()
    
    for coordinate in coordinates:
        # print(f"coordinate : {coordinate}")
        coordinates_msg.data.extend(coordinate)
    
    publisher.publish(coordinates_msg)
  


def test(model, dset, _cfg, logger, out_path_root, coordinates_publisher):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    # ori_voxels_path = "/home/melodic/jetsonNX/Aerial-Walker/src/oc_navigation/plan_manage/raw_data/ori_voxels"
    model = model.to(device=device)
    logger.info('=> Passing the network on the test set...')
    model.eval()
    inv_remap_lut = dset.dataset.get_inv_remap_lut()

    start_time = time.time()
    inference_time = []
   
    with torch.no_grad():
        for t, (data, indices) in enumerate(dset):
            data = dict_to(data, device, dtype)

            # Record the inference start time
            inference_start_time = time.time()

            scores = model(data)
            
            # Record the inference end time
            inference_end_time = time.time()
            
            # Log the inference time of each sample
            inference_time.append(inference_end_time - inference_start_time)

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
                
                # score = np.moveaxis(score, [0, 1, 2], [0, 2, 1]).reshape(-1).astype(np.uint16)
                # score = inv_remap_lut[score].astype(np.uint16)
                
              
                # filename, extension = os.path.splitext(os.path.basename(input_filename))
                # out_filename = os.path.join(out_path_root, 'predictions', filename + '.label')
                # _create_directory(os.path.dirname(out_filename))
                # score.tofile(out_filename)
                #shutil.copy(input_filename, ori_voxels_path)
                os.remove(input_filename)
                # curr_index += 1
  
    
    return inference_time


def main():
    rospy.init_node("inference_node")
    #Create the publisher using a specific ROS message type and topic
    coordinates_publisher = rospy.Publisher('/non_intersection_coordinates', Float64MultiArray, queue_size=1000) 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    torch.backends.cudnn.enabled = True
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
    wait_time = 0.4  # Seconds to wait before checking the dataset folder again
    train_batch_size = 1  # Set your desired batch_size here
    while not rospy.is_shutdown(): 
        rospy.sleep(wait_time)
        dataset = get_dataset(_cfg)['test']
             
        # dataset = None
        
        # while dataset is None:
        #     dataset_files = os.listdir(dataset_f)
            
            
        #     # Check if the dataset folder has sufficient data (files) for the batch size
        #     if len(dataset_files) >= train_batch_size:
        #         dataset = get_dataset(_cfg)['test']
        #         break
        #     else:
        #         rospy.loginfo("Waiting for dataset folder to accumulate sufficient files.")
        #         rospy.sleep(wait_time)
        # dataset = get_dataset(_cfg)['test']
        logger.info('=> Loading network architecture...')
        model = get_model(_cfg, dataset.dataset)
        
        logger.info('=> Loading network weights...')
        model = model.to(device=device)
        model = checkpoint.load_model(model, weights_f, logger)
        rate = rospy.Rate(10)  
        inference_time = test(model, dataset, _cfg, logger, out_path_root, coordinates_publisher)  
        logger.info('=> ============ Network Test Done ============')
        average_inference_time = np.sum(inference_time) / 1.0
        fps = 1 / average_inference_time
        logger.info('Inference time per frame is %.6f seconds\n' % average_inference_time)
        logger.info('FPS: %.2f\n' % fps)
        rate.sleep()

if __name__ == '__main__':
    main()