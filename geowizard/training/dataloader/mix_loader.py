# A reimplemented version in public environments by Xiao Fu and Mu Hu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import cv2

from dataloader.utils import read_text_lines
from dataloader.file_io import *

import pickle
import json
from skimage import io, transform
import numpy as np
import glob
import tqdm
from PIL import Image
import torch
from imgaug import augmenters as iaa

class MixDataset(Dataset):
    def __init__(self, data_dir,
                 load_pseudo_gt=False,
                 transform=None):
        super(MixDataset, self).__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.img_size = (576, 768)
    
        load_datasets = ['HyperSim', 'replica', '3d_ken_burns', 'simulation_disparity', 'objaverse']
        self.samples = []

        # 1. load hypersim dataset
        if 'HyperSim' in load_datasets:
            data_dir = os.path.join(self.data_dir, 'Hypersim', 'annotations', 'annos_all.json')

            with open(data_dir) as f:
                datas = json.load(f)['files']

            for data in tqdm.tqdm(datas, desc='Loading HyperSim'):
                meta_data = data['meta_data']
                with open(os.path.join(self.data_dir, data['meta_data']), 'rb') as f: meta_data = pickle.load(f)

                if meta_data['is_complete']['state'] == False:
                    if meta_data['is_complete']['is_crop'] == False:
                        continue
        
                sample = dict()
                sample['dataset'] = 'hypersim'
                sample['rgb'] = os.path.join(self.data_dir, meta_data['rgbs']['rgb_tonemap'])
                sample['depth'] = os.path.join(self.data_dir, meta_data['depth'])
                sample['normal'] = os.path.join(self.data_dir, meta_data['normal'])
                sample['cam_in'] = meta_data['cam_in']
                sample['metric_scale'] = 1.0

                # crop valid region in incomplete image
                sample['is_complete'] = meta_data['is_complete']['state']
                if sample['is_complete'] == False:
                    sample['crop_size'] = meta_data['is_complete']['crop_size']

                # data augmentation args
                sample['RandomHorizontalFlip'] = 0.4
                sample['distortion_prob'] = 0.05
                sample['to_gray_prob'] = 0.1

                self.samples.append(sample)


        # 2. load replica dataset
        if 'replica' in load_datasets:
            data_dir = os.path.join(self.data_dir, 'replica', 'annotations', 'annos_all.json')
            with open(data_dir) as f:
                datas = json.load(f)['files']

            for data in tqdm.tqdm(datas, desc='Loading Replica'):
                meta_data = data['meta_data']
                with open(os.path.join(self.data_dir, data['meta_data']), 'rb') as f: meta_data = pickle.load(f)
            
                sample = dict()
                sample['dataset'] = 'replica'
                sample['rgb'] = os.path.join(self.data_dir, meta_data['rgb'])
                sample['depth'] = os.path.join(self.data_dir, meta_data['depth'])
                sample['normal'] = os.path.join(self.data_dir, meta_data['normal'])
                sample['cam_in'] = meta_data['cam_in']
                sample['metric_scale'] = 512.0

                threshold = 50
                if meta_data['is_complete'] == False:
                    if meta_data['invalid_num'] > threshold:
                        continue

                # data augmentation args
                sample['RandomHorizontalFlip'] = 0.4
                sample['distortion_prob'] = 0.05
                sample['to_gray_prob'] = 0.1

                self.samples.append(sample)


        # 3. load 3d_ken_burns dataset
        if '3d_ken_burns' in load_datasets:
            data_dir = os.path.join(self.data_dir, '3d_ken_burns', 'annotations', 'annos_all.json')
            with open(data_dir) as f:
                datas = json.load(f)['files']

            for data in tqdm.tqdm(datas, desc='Loading 3d_ken_burns'):
                meta_data = data['meta_data']

                with open(os.path.join(self.data_dir, data['meta_data']), 'rb') as f: meta_data = pickle.load(f)

                sample = dict()
                sample['dataset'] = '3d_ken_burns'
                sample['rgb_l'] = os.path.join(self.data_dir, meta_data['rgb'])
                sample['rgb_r'] = os.path.join(self.data_dir, meta_data['rgb_right'])
                sample['depth_l'] = os.path.join(self.data_dir, meta_data['depth'])
                sample['depth_r'] = os.path.join(self.data_dir, meta_data['depth_right'])
                sample['normal_l'] = os.path.join(self.data_dir, meta_data['normal'])
                sample['normal_r'] = os.path.join(self.data_dir, meta_data['normal_right'])
                sample['cam_in'] = meta_data['cam_in']
                sample['metric_scale'] = 100.0

                # data augmentation args
                sample['RandomHorizontalFlip'] = 0.4
                sample['distortion_prob'] = 0.1
                sample['to_gray_prob'] = 0.2

                self.samples.append(sample)


        # 4. load simulation_disparity dataset
        if 'simulation_disparity' in load_datasets:
            data_dir = os.path.join(self.data_dir, 'simulation_disparity',  'annos_all.json')
            with open(data_dir) as f:
                datas = json.load(f)['files']

            for data in tqdm.tqdm(datas, desc='Loading simulation_disparity'):
                meta_data = data['meta_data']

                sample = dict()
                sample['dataset'] = 'simulation_disparity'
                sample['rgb'] = os.path.join(self.data_dir, meta_data['rgb'])
                sample['depth'] = os.path.join(self.data_dir, meta_data['depth'])
                sample['normal'] = os.path.join(self.data_dir, meta_data['normal'])
                sample['cam_in'] = meta_data['cam_in']
                sample['metric_scale'] = 128.0

                # data augmentation args
                sample['RandomHorizontalFlip'] = 0.4
                sample['distortion_prob'] = 0.1
                sample['to_gray_prob'] = 0.2

                self.samples.append(sample)


        # 5. load objaverse dataset
        if 'objaverse' in load_datasets:
            data_dir = os.path.join(self.data_dir, 'objaverse',  'annos_all.json')
            with open(data_dir) as f:
                datas = json.load(f)

            for data in tqdm.tqdm(datas, desc='Loading objaverse'):

                sample = dict()
                sample['dataset'] = 'objaverse'
                sample['meta_dir'] = os.path.join(self.data_dir, 'objaverse', data)

                # data augmentation args
                sample['RandomHorizontalFlip'] = 0.4
                sample['distortion_prob'] = 0.05
                sample['to_gray_prob'] = 0.1

                self.samples.append(sample)


    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        H, W = self.img_size

        # ----------------- Load Different Datasets -----------------

        # HyperSim
        if sample_path['dataset'] == 'hypersim':
            sample['domain'] = torch.Tensor([1., 0., 0.]) # indoor

            sample['rgb'] = read_img(sample_path['rgb'])  # [H, W, 3]
            sample['depth'], sample['normal'] = read_depth_normal_hypersim(sample_path['depth'], sample_path['normal'], sample_path['cam_in'], sample_path['metric_scale'])

            H_ori, W_ori = sample['rgb'].shape[:2]
 
            # crop valid region in incomplete image
            if sample_path['is_complete'] == False:

                H_start, H_end, W_start, W_end = sample_path['crop_size']
                sample['rgb'] = sample['rgb'][H_start:H_end, W_start:W_end]
                sample['depth'] = sample['depth'][H_start:H_end, W_start:W_end]
                sample['normal'] = sample['normal'][H_start:H_end, W_start:W_end]
                assert np.isnan(sample['depth']).sum() == 0 

                up_scale = 1.2
                H_ori, W_ori = sample['rgb'].shape[:2]
                up_size = (int(W_ori * up_scale), int(H_ori * up_scale))
                sample['rgb'] = cv2.resize(sample['rgb'], up_size, interpolation = cv2.INTER_CUBIC)
                sample['depth'] = cv2.resize(sample['depth'], up_size, interpolation = cv2.INTER_NEAREST)
                sample['depth'] /= up_scale
                sample['normal'] = cv2.resize(sample['normal'], up_size, interpolation = cv2.INTER_NEAREST)

                H_ori, W_ori = sample['rgb'].shape[:2]
                assert H_ori >= H, W_ori >= W


        # replica
        if sample_path['dataset'] == 'replica':
            sample['domain'] = torch.Tensor([1., 0., 0.]) # indoor

            sample['rgb'] = read_img(sample_path['rgb'])  # [H, W, 3]
            sample['depth'], sample['normal'], invalid_mask = read_depth_normal_replica(sample_path['depth'], sample_path['normal'], sample_path['cam_in'], sample_path['metric_scale'])

            up_scale = 1.5
            H_ori, W_ori = sample['rgb'].shape[:2]
            up_size = (int(W_ori * up_scale), int(H_ori * up_scale))

            sample['rgb'] = cv2.resize(sample['rgb'], up_size, interpolation = cv2.INTER_CUBIC)
            sample['depth'] = cv2.resize(sample['depth'], up_size, interpolation = cv2.INTER_NEAREST)
            sample['depth'] /= up_scale
            sample['normal'] = cv2.resize(sample['normal'], up_size, interpolation = cv2.INTER_NEAREST)
            H_ori, W_ori = sample['rgb'].shape[:2]


        # 3d ken burns
        if sample_path['dataset'] == '3d_ken_burns':
            sample['domain'] = torch.Tensor([0., 1., 0.]) # outdoor

            if np.random.random() < 0.5:
                rgb_path = sample_path['rgb_l']
                depth_path = sample_path['depth_l']
                normal_path = sample_path['normal_l']
            else:
                rgb_path = sample_path['rgb_r']
                depth_path = sample_path['depth_r']
                normal_path = sample_path['normal_r']

            sample['rgb'] = read_img(rgb_path)  # [H, W, 3]
            sample['depth'], sample['normal'], invalid_mask = read_depth_normal_kenburns(depth_path, normal_path, sample_path['cam_in'], sample_path['metric_scale'])

            up_scale = 1.5
            H_ori, W_ori = sample['rgb'].shape[:2]
            up_size = (int(W_ori * up_scale), int(H_ori * up_scale))

            sample['rgb'] = cv2.resize(sample['rgb'], up_size, interpolation = cv2.INTER_CUBIC)
            sample['depth'] = cv2.resize(sample['depth'], up_size, interpolation = cv2.INTER_NEAREST)
            sample['depth'] /= up_scale
            sample['normal'] = cv2.resize(sample['normal'], up_size, interpolation = cv2.INTER_NEAREST)
            H_ori, W_ori = sample['rgb'].shape[:2]


        # simulation_disparity
        if sample_path['dataset'] == 'simulation_disparity':

            sample['domain'] = torch.Tensor([0., 1., 0.]) # outdoor
            
            sample['rgb'] = read_img(sample_path['rgb'])  # [H, W, 3]
            sample['depth'], sample['normal'] = read_depth_normal_simulation_disparity(sample_path['depth'], sample_path['normal'], sample_path['cam_in'], sample_path['metric_scale'])

            up_scale = 0.4
            H_ori, W_ori = sample['rgb'].shape[:2]
            up_size = (int(W_ori * up_scale), int(H_ori * up_scale))

            sample['rgb'] = cv2.resize(sample['rgb'], up_size, interpolation = cv2.INTER_CUBIC)
            sample['depth'] = cv2.resize(sample['depth'], up_size, interpolation = cv2.INTER_NEAREST)
            sample['depth'] /= up_scale
            sample['normal'] = cv2.resize(sample['normal'], up_size, interpolation = cv2.INTER_NEAREST)
            H_ori, W_ori = sample['rgb'].shape[:2]
        
        # objaverse 
        if sample_path['dataset'] == 'objaverse':
            sample['domain'] = torch.Tensor([0., 0., 1.]) # object

            views = ['00000', '00010', '00020', '00030']
            idx = np.random.randint(0,4)
            view = views[idx]

            # downloading bug
            try:
                json_file = os.path.join(sample_path['meta_dir'], view, view+'.json')
                c2w = read_camera_matrix_single(json_file)
            except:
                with open(f'fail_objaverse.txt', 'a+') as f:
                    f.write(sample_path['meta_dir'])
                    f.write('\n')
                if idx == 0:
                    idx = 3
                else:
                    idx = idx - 1
                view = views[idx]
                json_file = os.path.join(sample_path['meta_dir'], view, view+'.json')
                c2w = read_camera_matrix_single(json_file)

            cam_dis = np.linalg.norm(c2w[:3, 3:], 2)
            near = 0.867
            near_distance = cam_dis - near
            normald_path = os.path.join(sample_path['meta_dir'], view, view+'_nd.exr')
            normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth = normald[...,3:]
            depth[depth<near_distance] = 0
            depth[depth==0] = 5.
            depth = depth[:,:,0]
            mask_depth = depth==5.

            normal = normald[...,:3]
            normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
            normal = normal / normal_norm
            normal = np.nan_to_num(normal,nan=-1.)
            world_normal = unity2blender(normal)
            normal = blender2midas(world_normal@ (c2w[:3,:3]))
            normal = normal / (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
            mask_normal = np.asarray(np.clip((normal+1.)/2. * 255, 0, 255), np.uint8)[...,-1] >= 127
            normal[:,:,0] *= -1.

            rgb_path = os.path.join(sample_path['meta_dir'], view, view+'.png')
            rgbd = read_img(rgb_path)
            rgb = rgbd[:,:,:3]
            alpha = rgbd[:,:,3][...,None]/255.
            bg_color = np.array([255.,255.,255.])
            rgb = (rgb * alpha + bg_color*(1-alpha)).astype(np.uint8)
            mask_rgb = rgbd[:,:,3]==0

            mask = mask_depth | mask_normal | mask_rgb
            depth[mask] = 5.
            normal[mask] = np.array([0.,0.,-1.])
            rgb[mask] = 255

            sample['depth'] = depth
            sample['normal'] = normal
            sample['rgb'] = rgb

            up_scale = 1.125
            H_ori, W_ori = sample['rgb'].shape[:2]
            up_size = (int(W_ori * up_scale), int(H_ori * up_scale))
            sample['rgb'] = cv2.resize(sample['rgb'], up_size, interpolation = cv2.INTER_CUBIC)
            sample['depth'] = cv2.resize(sample['depth'], up_size, interpolation = cv2.INTER_NEAREST)
            sample['depth'] /= up_scale
            sample['normal'] = cv2.resize(sample['normal'], up_size, interpolation = cv2.INTER_NEAREST)

            rgb_padding = np.full((576,96,3), 255, dtype=np.uint8)
            sample['rgb'] = np.concatenate((rgb_padding, sample['rgb'], rgb_padding), axis=1)
            depth_padding = np.full((576,96), sample['depth'][0,0], dtype=np.float32)
            sample['depth'] = np.concatenate((depth_padding, sample['depth'], depth_padding), axis=1)
            normal_temp = np.zeros((576,96,3), dtype=np.float32)
            normal_temp[:,:,-1] = -1.
            sample['normal'] = np.concatenate((normal_temp, sample['normal'], normal_temp), axis=1)
            H_ori, W_ori = sample['rgb'].shape[:2]


        # ----------------- Data Augmentation -----------------

        # 1. Random Crop
        if H_ori >= H and W_ori >= W:
            H_start, W_start = np.random.randint(0, H_ori-H+1), np.random.randint(0, W_ori-W+1)
            sample['rgb'] = sample['rgb'][H_start:H_start + H, W_start:W_start + W]
            sample['depth'] = sample['depth'][H_start:H_start + H, W_start:W_start + W]
            sample['normal'] = sample['normal'][H_start:H_start + H, W_start:W_start + W]

        # 2. Random Horizontal Flip
        if np.random.random() < sample_path['RandomHorizontalFlip']:
            sample['rgb'] = np.copy(np.fliplr(sample['rgb']))
            sample['depth'] = np.copy(np.fliplr(sample['depth']))
            sample['normal'] = np.copy(np.fliplr(sample['normal']))
            sample['normal'][:,:,0] *= -1.

        # 3. Photometric Distortion
        to_gray_prob = sample_path['to_gray_prob']
        distortion_prob = sample_path['distortion_prob']
        brightness_beta = np.random.uniform(-32, 32)
        contrast_alpha = np.random.uniform(0.5, 1.5)
        saturate_alpha = np.random.uniform(0.5, 1.5)
        rand_hue = np.random.randint(-18, 18)

        brightness_do = np.random.random() < distortion_prob
        contrast_do = np.random.random() < distortion_prob
        saturate_do = np.random.random() < distortion_prob
        rand_hue_do = np.random.random() < distortion_prob

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = 0 if np.random.random() > 0.5 else 1
        if np.random.random() < to_gray_prob:
            sample['rgb'] = iaa.Grayscale(alpha=(0.8, 1.0))(image=sample['rgb'])
        else:
            # random brightness
            if brightness_do:
                alpha, beta = 1.0, brightness_beta
                sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

            if mode == 0:
                if contrast_do:
                    alpha, beta = contrast_alpha, 0.0
                    sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

            # random saturation
            if saturate_do:
                img = cv2.cvtColor(sample['rgb'][:,:,::-1], cv2.COLOR_BGR2HSV)
                alpha, beta = saturate_alpha, 0.0
                img[:,:,1] = np.clip((img[:,:,1].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)
                sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:,:,::-1]

            # random hue
            if rand_hue_do:
                img = cv2.cvtColor(sample['rgb'][:,:,::-1], cv2.COLOR_BGR2HSV)
                img[:, :, 0] = (img[:, :, 0].astype(int) + rand_hue) % 180
                sample['rgb'] = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)[:,:,::-1]

            # random contrast
            if mode == 1:
                if contrast_do:
                    alpha, beta = contrast_alpha, 0.0
                    sample['rgb'] = np.clip((sample['rgb'].astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)

        # 4. To Tensor
        sample['rgb'] = (torch.from_numpy(np.transpose(sample['rgb'].copy(), (2, 0, 1))) / 255.) * 2.0 - 1.0  # [3, H, W]
        sample['depth'] = torch.from_numpy(sample['depth'][None].copy())  # [1, H, W]
        sample['normal'] = torch.from_numpy(np.transpose(sample['normal'].copy(), (2, 0, 1)))  # [3, H, W]

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size
