# A reimplemented version in public environments by Xiao Fu and Mu Hu

from __future__ import division
from genericpath import samefile
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as F
import random
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample



class ToTensor(object):
    """Convert numpy array to torch tensor"""
    def __call__(self, sample):
        img = np.transpose(sample['rgb'], (2, 0, 1))  # [3, H, W]
        sample['rgb'] = torch.from_numpy(img) / 255.
        sample['depth'] = torch.from_numpy(sample['depth'])  # [H, W]
        sample['normal'] = torch.from_numpy(sample['normal'])  # [3, H, W]

        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['img']
        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class RandomCrop(object):
    def __init__(self, img_height, img_width, validate=False):
        self.img_height = img_height
        self.img_width = img_width
        self.validate = validate

    def __call__(self, sample):
        

        ori_height, ori_width = sample['rgb'].shape[:2]
        if self.img_height > ori_height or self.img_width > ori_width:
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width

            assert top_pad >= 0 and right_pad >= 0

            sample['rgb'] = np.lib.pad(sample['rgb'],
                                        ((top_pad, 0), (0, right_pad), (0, 0)),
                                        mode='constant',
                                        constant_values=0)

            if 'depth' in sample.keys():
                sample['depth'] = np.lib.pad(sample['depth'],
                                            ((top_pad, 0), (0, right_pad)),
                                            mode='constant',
                                            constant_values=0)

        else:
            assert self.img_height <= ori_height and self.img_width <= ori_width
            # Training: random crop
            if not self.validate:

                self.offset_x = np.random.randint(ori_width - self.img_width + 1)

                start_height = 0
                assert ori_height - start_height >= self.img_height

                self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)

            # Validatoin, center crop
            else:
                self.offset_x = (ori_width - self.img_width) // 2
                self.offset_y = (ori_height - self.img_height) // 2

            sample['rgb'] = self.crop_img(sample['rgb'])
            if 'depth' in sample.keys():
                sample['depth'] = self.crop_img(sample['depth'])
        
        return sample

    def crop_img(self, img):
        return img[self.offset_y:self.offset_y + self.img_height,
               self.offset_x:self.offset_x + self.img_width]

import matplotlib.pyplot as plt

class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.09:
            sample['rgb'] = np.copy(np.flipud(sample['rgb']))
            sample['depth'] = np.copy(np.flipud(sample['depth']))
            sample['normal'] = np.copy(np.flipud(sample['normal']))

        return sample


class ToPILImage(object):

    def __call__(self, sample):
        sample['rgb'] = Image.fromarray(sample['rgb'].astype('uint8'))

        return sample


class ToNumpyArray(object):

    def __call__(self, sample):
        sample['rgb'] = np.array(sample['rgb']).astype(np.float32)

        return sample


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['rgb'] = F.adjust_contrast(sample['rgb'], contrast_factor)

        return sample


class RandomGamma(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['rgb'] = F.adjust_gamma(sample['rgb'], gamma)
        
        return sample


class RandomBrightness(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample['rgb'] = F.adjust_brightness(sample['rgb'], brightness)

        return sample


class RandomHue(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['rgb'] = F.adjust_hue(sample['rgb'], hue)

        return sample


class RandomSaturation(object):

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['rgb'] = F.adjust_saturation(sample['rgb'], saturation)
        return sample


class RandomColor(object):

    def __call__(self, sample):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        sample = ToPILImage()(sample)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample = t(sample)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample = t(sample)

        sample = ToNumpyArray()(sample)

        return sample