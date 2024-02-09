from torch.utils.data import Dataset
import glob
import os
import numpy as np
import yaml
import random
import sys

import network.data.io_data as SemanticKittiIO


class SemanticKITTI_dataloader(Dataset):

  def __init__(self, dataset, phase):
    '''

    :param dataset: The dataset configuration (data augmentation, input encoding, etc)
    :param phase_tag: To differentiate between training, validation and test phase
    '''

    yaml_path, _ = os.path.split(os.path.realpath(__file__))
    self.dataset_config = yaml.safe_load(open(os.path.join(yaml_path, 'semantic-kitti.yaml'), 'r'))
    self.nbr_classes = self.dataset_config['nbr_classes']
    self.grid_dimensions = self.dataset_config['grid_dims']   # [W, H, D]
    self.remap_lut = self.get_remap_lut()
    self.rgb_mean = np.array([0.34749558, 0.36745213, 0.36123651])  # images mean:  [88.61137282 93.70029365 92.11530949]
    self.rgb_std = np.array([0.30599035, 0.3129534 , 0.31933814])   # images std:  [78.02753826 79.80311686 81.43122464]
    self.root_dir = dataset['ROOT_DIR']
    self.modalities = dataset['MODALITIES']
    self.extensions = {'3D_OCCUPANCY': '.bin', '3D_LABEL': '.label', '3D_OCCLUDED': '.occluded',
                       '3D_INVALID': '.invalid'}
    self.data_augmentation = {'FLIPS': dataset['AUGMENTATION']['FLIPS']}

    self.filepaths = {}
    self.phase = phase
    self.class_frequencies = np.array([5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
                                       6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
                                       2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
                                       2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
                                       2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05])

    self.split = {'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10], 'val': [8],
                  'test': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]}

    for modality in self.modalities:
      if self.modalities[modality]:
        self.get_filepaths(modality)

    # if self.phase != 'test':
    #   self.check_same_nbr_files()

    self.nbr_files = len(self.filepaths['3D_OCCUPANCY'])  # TODO: Pass to something generic

    return
  
  # def next_file_batches(self, dataset_dir, file_pattern, batch_size=8):
  #     filepaths = sorted(glob.glob(os.path.join(dataset_dir, file_pattern)))
  #     start_index = 0
  #     while start_index < len(filepaths):
  #         yield filepaths[start_index:start_index + batch_size]
  #         start_index += batch_size
          
        
  def get_filepaths(self, modality):
      '''
      Set modality filepaths with split according to phase (train, val, test)
      '''

      if modality == '3D_OCCUPANCY':
          self.filepaths['3D_OCCUPANCY'] = []  # Clear the file paths before each iteration
          
          dataset_dir =  "/root/AGRNav/src/planner/plan_manage/raw_data/voxels"
          file_pattern = '*.bin'

          # Get all file paths in the directory and sort them
          filepaths = sorted(glob.glob(os.path.join(dataset_dir, file_pattern)))
          # import pdb
          # pdb.set_trace()
          # Only consider the first batch_size files
          batch_size = 6
          self.filepaths['3D_OCCUPANCY'] = filepaths[:batch_size]
          
          # Remove the processed files from the list
          #filepaths = filepaths[batch_size:]

           
         

      # Combine the dataset path, file pattern (e.g., '*.bin') and sort the resulting list
      #self.filepaths['3D_OCCUPANCY'] = sorted(glob(os.path.join(dataset_dir, file_pattern)))
      #print(self.filepaths['3D_OCCUPANCY'])
      


  def check_same_nbr_files(self):
    '''
    Set modality filepaths with split according to phase (train, val, test)
    '''

    # TODO: Modify for nested dictionaries...
    for i in range(len(self.filepaths.keys()) - 1):
      length1 = len(self.filepaths[list(self.filepaths.keys())[i]])
      length2 = len(self.filepaths[list(self.filepaths.keys())[i+1]])
      assert length1 == length2, 'Error: {} and {} not same number of files'.format(list(self.filepaths.keys())[i],
                                                                                    list(self.filepaths.keys())[i+1])
    return

  def __getitem__(self, idx):

    data = {}

    do_flip = 0
    if self.data_augmentation['FLIPS'] and self.phase == 'train':
      do_flip = random.randint(0, 3)

    for modality in self.modalities:
      if (self.modalities[modality]) and (modality in self.filepaths):
        data[modality] = self.get_data_modality(modality, idx, do_flip)

    return data, idx

  def get_data_modality(self, modality, idx, flip):

    if modality == '3D_OCCUPANCY':
      OCCUPANCY = SemanticKittiIO._read_occupancy_SemKITTI(self.filepaths[modality][idx])
      # print(OCCUPANCY.reshape([self.grid_dimensions[0],
      #                                            self.grid_dimensions[2],
      #                                            self.grid_dimensions[1]]).shape)
      
      OCCUPANCY = np.moveaxis(OCCUPANCY.reshape([self.grid_dimensions[0],
                                                 self.grid_dimensions[2],
                                                 self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
      OCCUPANCY = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCUPANCY)
      return OCCUPANCY[None, :, :, :]

    elif modality == '3D_LABEL':
      LABEL_1_1 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_1', idx))
      LABEL_1_2 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_2', idx))
      LABEL_1_4 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_4', idx))
      LABEL_1_8 = SemanticKittiIO.data_augmentation_3Dflips(flip, self.get_label_at_scale('1_8', idx))
      return {'1_1': LABEL_1_1, '1_2': LABEL_1_2, '1_4': LABEL_1_4, '1_8': LABEL_1_8}

    elif modality == '3D_OCCLUDED':
      OCCLUDED = SemanticKittiIO._read_occluded_SemKITTI(self.filepaths[modality][idx])
      OCCLUDED = np.moveaxis(OCCLUDED.reshape([self.grid_dimensions[0],
                                               self.grid_dimensions[2],
                                               self.grid_dimensions[1]]), [0, 1, 2], [0, 2, 1])
      OCCLUDED = SemanticKittiIO.data_augmentation_3Dflips(flip, OCCLUDED)
      return OCCLUDED

    else:
      assert False, 'Specified modality not found'

  def get_label_at_scale(self, scale, idx):

    scale_divide = int(scale[-1])
    INVALID = SemanticKittiIO._read_invalid_SemKITTI(self.filepaths['3D_INVALID'][scale][idx])
    LABEL = SemanticKittiIO._read_label_SemKITTI(self.filepaths['3D_LABEL'][scale][idx])
    if scale == '1_1':
      LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
    LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
    LABEL = np.moveaxis(LABEL.reshape([int(self.grid_dimensions[0] / scale_divide),
                                       int(self.grid_dimensions[2] / scale_divide),
                                       int(self.grid_dimensions[1] / scale_divide)]), [0, 1, 2], [0, 2, 1])

    return LABEL

  def read_semantics_config(self, data_path):

    # get number of interest classes, and the label mappings
    DATA = yaml.safe_load(open(data_path, 'r'))
    self.class_strings = DATA["labels"]
    self.class_remap = DATA["learning_map"]
    self.class_inv_remap = DATA["learning_map_inv"]
    self.class_ignore = DATA["learning_ignore"]
    self.n_classes = len(self.class_inv_remap)

    return

  def get_inv_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(self.dataset_config['learning_map_inv'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map_inv'].keys())] = list(self.dataset_config['learning_map_inv'].values())

    return remap_lut

  def get_remap_lut(self):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(self.dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(self.dataset_config['learning_map'].keys())] = list(self.dataset_config['learning_map'].values())

    # in completion we have to distinguish empty and invalid voxels.
    # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    return remap_lut

  def __len__(self):
    """
    Returns the length of the dataset
    """
    # Return the number of elements in the dataset
    return self.nbr_files

