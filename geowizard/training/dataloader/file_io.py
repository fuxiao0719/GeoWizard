# A reimplemented version in public environments by Xiao Fu


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys
import torch
import cv2
import h5py
from utils.depth2normal import *
import json

def read_img(filename):
    img = np.array(Image.open(filename))
    return img

def read_hdf5(filename):
    with h5py.File(filename, "r") as f:  file = f["dataset"][:]
    return file

def distance2depth(npyDistance:np.ndarray, fitFocal:float):
    '''
    fitFocal is one of fx or fy in K
    '''
    intHeight, intWidth = npyDistance.shape
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fitFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fitFocal
    return npyDepth

def align_normal(normal, depth, K, H, W):
    '''
    Orientation of surface normals in hypersim is not always consistent
    see https://github.com/apple/ml-hypersim/issues/26
    '''
    # inv K
    K = np.array([[K[0], 0 ,K[2]], 
                    [0, K[1], K[3]], 
                    [0, 0, 1]])
    inv_K = np.linalg.inv(K)
    # reprojection depth to camera points

    xy = creat_uv_mesh(H, W)
    points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
    points = depth * points
    points = points.transpose((1,2,0))

    # align normal
    orient_mask = np.sum(normal * points, axis=2) > 0
    normal[orient_mask] *= -1

    return normal

def creat_uv_mesh(H, W):
    y, x = np.meshgrid(np.arange(0, H, dtype=np.float), np.arange(0, W, dtype=np.float), indexing='ij')
    meshgrid = np.stack((x,y))
    ones = np.ones((1,H*W), dtype=np.float)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)

def read_depth_normal_hypersim(depth_path, normal_path, K, metric_scale):

    depth = read_hdf5(depth_path).astype(np.float32)
    depth[depth>60000] = 0
    depth = depth / metric_scale

    normal = read_hdf5(normal_path).astype(np.float32)
    H, W = normal.shape[:2]
    # convert (x right, y up, z backward) to conventional (x right, y down, z forward)
    normal[:,:,1:] *= -1
    normal = align_normal(normal, depth, K, H, W)
    normal /= (np.linalg.norm(normal, ord=2, axis=2, keepdims=True) + 1e-5)
    return depth, normal

def read_depth_normal_replica(depth_path, normal_path, K, metric_scale):

    depth = cv2.imread(depth_path, -1)
    depth[depth>60000] = 0
    depth = depth / metric_scale

    with open(normal_path, 'rb') as f:
        normal = Image.open(f)
        normal = np.array(normal.convert(normal.mode), dtype=np.uint8)
    invalid_mask = np.all(normal == 128, axis=2)
    normal = normal.astype(np.float64) / 255.0 * 2 - 1
    normal[invalid_mask, :] = 0
    normal /= (np.linalg.norm(normal, ord=2, axis=2, keepdims=True) + 1e-5)

    depth[invalid_mask] = 10.
    normal[invalid_mask] = np.array([0., 0., -1.])
    
    return depth, normal, invalid_mask


def read_depth_normal_virtual_kitti(depth_path, normal_path, K, metric_scale):

    depth = cv2.imread(depth_path, -1)
    depth[depth>(150 * metric_scale)] = 0
    depth = depth / metric_scale

    far_plane = 80
    invalid_mask = (depth == 0.)
    depth[invalid_mask] = far_plane
    depth[depth>far_plane] = far_plane

    # focal_length = K[:2]
    # depth2normal = surface_normal_from_depth(torch.from_numpy(depth[None,None]).to(torch.float32).cuda(), torch.Tensor(focal_length).cuda())
    # depth2normal = depth2normal.cpu().numpy()[0].transpose(1,2,0)
    # depth2normal /= (np.linalg.norm(depth2normal, axis=-1, ord=2, keepdims=True) + 1e-5)
    
    normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    normal = normal.astype(np.float64) / 255.0 * 2 - 1
    normal = normal[:,:,::-1]
    normal = normal / (np.linalg.norm(normal, ord=2, axis=2, keepdims=True) + 1e-5)

    return depth, normal, invalid_mask


def read_depth_normal_simulation_disparity(depth_path, normal_path, K, metric_scale):

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth / metric_scale

    # far-range clip
    far_plane = 510
    mask = depth>far_plane
    depth[mask] = far_plane

    normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    normal = normal.astype(np.float64) / 255.0 * 2 - 1
    normal = normal[:,:,::-1]
    normal = normal / (np.linalg.norm(normal, ord=2, axis=2, keepdims=True) + 1e-5)
    normal[mask, :] = np.array([0., 0., -1.])

    return depth, normal


def read_depth_normal_kenburns(depth_path, normal_path, K, metric_scale):

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth[depth>50000] = 0.
    depth = depth / metric_scale
    invalid_mask = depth == 0.

    # far-range clip
    far_plane = 80
    depth[invalid_mask] = far_plane
    depth[depth>far_plane] = far_plane

    normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    normal[~invalid_mask] /= (np.linalg.norm(normal, axis=-1, ord=2)[~invalid_mask][...,None] + 1e-5)
    normal[:,:,1] *= -1
    normal[invalid_mask, :] = np.array([0., 0., -1.])

    return depth, normal, invalid_mask


def depth2disp(left_depth_path):
    left_depth = np.array(Image.open(left_depth_path))/(10000)
    focal_length = 768.16058349609375
    baseline = 0.06
    left_disp = focal_length*baseline /left_depth
    return left_disp


def unity2blender(normal):
    normal_clone = normal.copy()
    normal_clone[...,0] = -normal[...,-1]
    normal_clone[...,1] = -normal[...,0]
    normal_clone[...,2] = normal[...,1]

    return normal_clone


def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''
    img[...,0] = -img[...,0]
    img[...,1] = -img[...,1]
    img[...,-1] = -img[...,-1]
    return img


def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    # NOTE that different from unity2blender experiments.
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = -np.array(json_content['y'])
    camera_matrix[:3, 2] = -np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])


    '''
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    '''

    return camera_matrix

def depth2disp_cv(left_depth_path):
    left_depth = cv2.imread(left_depth_path, cv2.IMREAD_UNCHANGED)/ 10000
    focal_length = 768.16058349609375
    baseline = 0.06
    left_disp = focal_length*baseline /left_depth
    return left_disp


def read_disp(filename, subset=False):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # KITTI
    elif filename.endswith('png'):
        disp = _read_kitti_disp(filename)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)


def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth


def read_occlusion_mid(filename):
    img = Image.open(filename)
    img_np = np.array(img)
    valid_mask_combine = (img_np<=128).astype(np.float)    
    return valid_mask_combine