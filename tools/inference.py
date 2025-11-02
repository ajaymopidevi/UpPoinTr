##############################################################
# % Author: Ajay Narasimha Mopidevi
# % Date:01/11/2025
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
import open3d as o3d
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single_ColoRadar(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    # read single point cloud
    pc_ndarray = IO.get(pc_file).astype(np.float32)
    # transform it according to the model 
    # In GT, 1st point is the seed/anchor point 
    # In Input, 1st point is the closest point to the seed/anchor point
    # Centering around seed/anchor point
    
    centroid = pc_ndarray[0]
    pc_ndarray = pc_ndarray - centroid
    # m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
    m = 25
    pc_ndarray = pc_ndarray / m

    transform = Compose([{
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    fine_points = ret[-1].squeeze(0).detach().cpu().numpy()

    # denormalize it to adapt for the original input
    fine_points = fine_points * m
    fine_points = fine_points + centroid

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        fine_pcd = o3d.geometry.PointCloud()
        fine_pcd.points = o3d.utility.Vector3dVector(np.array(fine_points))
        o3d.io.write_point_cloud(target_path+ '_fine.pcd', fine_pcd, True, True)
        
        if args.save_vis_img:
            input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
            dense_img = misc.get_ptcloud_img(fine_points)
            cv2.imwrite(os.path.join(target_path, '_input.jpg'), input_img)
            cv2.imwrite(os.path.join(target_path, '_fine.jpg'), dense_img)
    
    return


def inference_single(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    # read single point cloud
    pc_ndarray = IO.get(pc_file).astype(np.float32)
    # transform it according to the model 
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # normalize it to fit the model on ShapeNet-55/34
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # denormalize it to adapt for the original input
        dense_points = dense_points * m
        dense_points = dense_points + centroid

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)

        np.save(os.path.join(target_path, 'fine.npy'), dense_points)
        if args.save_vis_img:
            input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
            dense_img = misc.get_ptcloud_img(dense_points)
            cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
            cv2.imwrite(os.path.join(target_path, 'fine.jpg'), dense_img)
    
    return

def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    if args.pc_root != '':
        pc_file_list = os.listdir(args.pc_root)
        if args.out_pc_root != '':
            os.makedirs(args.out_pc_root, exist_ok=True)
        start = time.time()
        for pc_file in pc_file_list:
            if config.dataset.train._base_['NAME'] == 'ColoRadar':
                inference_single_ColoRadar(base_model, pc_file, args, config, root=args.pc_root)
            else:
                inference_single(base_model, pc_file, args, config, root=args.pc_root)
        end = time.time()
        print("Runtime:", (end-start)*(10**3), " ms")
    else:
        start = time.time()
        if config.dataset.train._base_['NAME'] == 'ColoRadar':
            inference_single_ColoRadar(base_model, args.pc, args, config)
        else:
            inference_single(base_model, args.pc, args, config)
        end = time.time()
        print("Runtime:", (end-start)*(10**3), " ms")

if __name__ == '__main__':
    main()