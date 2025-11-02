import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import data_transforms
import logging

@DATASETS.register_module()
class ColoRadar(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.gt_path = config.GT_PATH
        self.input_path = config.INPUT_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        
    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')

        
    
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('run')[0]
            model_id = line.split('run')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'input_path': self.input_path % (line),
                'gt_path' : self.gt_path % (line)
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, input, gt):
        """ pc: NxC, return NxC """
        # Based on patch creation, centroid is 1st point
        # All points within a fixed distance are considered
        # Fixed distance ensures normalization without any externals points from GT
        centroid = input[0]
        gt = gt - centroid
        m = 25.0
        gt = gt / m
        input = input - centroid
        input = input/m
        input = np.clip(input, -1, 1)
        return input, gt
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = {}
        
        for ri in ['input', 'gt']:
            file_path = sample['%s_path' % ri]
            points = IO.get(file_path).astype(np.float32)
            data[ri] = points
        data['input'], data['gt'] = self.pc_norm(data['input'], data['gt'])
        
        assert data['gt'].shape[0] == self.npoints
        
        if self.transforms is not None:
            data = self.transforms(data)
        
        return sample['taxonomy_id'], sample['model_id'] , (data['input'], data['gt'])



    def __len__(self):
        return len(self.file_list)
