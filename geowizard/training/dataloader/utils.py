# A reimplemented version in public environments by Xiao Fu and Mu Hu

import os
import sys
import json
import torch
from glob import glob
import logging

import numpy as np


def find_occ_mask(disp_left, disp_right):
    """
    find occlusion map
    1 indicates occlusion
    disp range [0,w]
    """
    w = disp_left.shape[-1]

    # # left occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    right_shifted = coord - disp_left

    # 1. negative locations will be occlusion
    occ_mask_l = right_shifted <= 0

    # 2. wrong matches will be occlusion
    right_shifted[occ_mask_l] = 0  # set negative locations to 0
    right_shifted = right_shifted.astype(np.int)
    disp_right_selected = np.take_along_axis(disp_right, right_shifted,
                                             axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_right_selected - disp_left) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_right_selected <= 0.0] = False
    wrong_matches[disp_left <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_l] = True  # apply case 1 occlusion to case 2
    occ_mask_l = wrong_matches

    # # right occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    left_shifted = coord + disp_right

    # 1. negative locations will be occlusion
    occ_mask_r = left_shifted >= w

    # 2. wrong matches will be occlusion
    left_shifted[occ_mask_r] = 0  # set negative locations to 0
    left_shifted = left_shifted.astype(np.int)
    disp_left_selected = np.take_along_axis(disp_left, left_shifted,
                                            axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_left_selected - disp_right) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_left_selected <= 0.0] = False
    wrong_matches[disp_right <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_r] = True  # apply case 1 occlusion to case 2
    occ_mask_r = wrong_matches

    return occ_mask_l, occ_mask_r



def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def save_command(save_path, filename='command_train.txt'):
    check_path(save_path)
    command = sys.argv
    save_file = os.path.join(save_path, filename)
    with open(save_file, 'w') as f:
        f.write(' '.join(command))


def save_args(args, filename='args.json'):
    args_dict = vars(args)
    check_path(args.checkpoint_dir)
    save_path = os.path.join(args.checkpoint_dir, filename)

    with open(save_path, 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)


def int_list(s):
    """Convert string to int list"""
    return [int(x) for x in s.split(',')]


def save_checkpoint(save_path, optimizer, aanet, epoch, num_iter,
                    epe, best_epe, best_epoch, filename=None, save_optimizer=True):
    # AANet
    aanet_state = {
        'epoch': epoch,
        'num_iter': num_iter,
        'epe': epe,
        'best_epe': best_epe,
        'best_epoch': best_epoch,
        'state_dict': aanet.state_dict()
    }
    aanet_filename = 'aanet_epoch_{:0>3d}.pth'.format(epoch) if filename is None else filename
    aanet_save_path = os.path.join(save_path, aanet_filename)
    torch.save(aanet_state, aanet_save_path)

    # Optimizer
    if save_optimizer:
        optimizer_state = {
            'epoch': epoch,
            'num_iter': num_iter,
            'epe': epe,
            'best_epe': best_epe,
            'best_epoch': best_epoch,
            'state_dict': optimizer.state_dict()
        }
        optimizer_name = aanet_filename.replace('aanet', 'optimizer')
        optimizer_save_path = os.path.join(save_path, optimizer_name)
        torch.save(optimizer_state, optimizer_save_path)


def load_pretrained_net(net, pretrained_path, return_epoch_iter=False, resume=False,
                        no_strict=False):
    if pretrained_path is not None:
        if torch.cuda.is_available():
            state = torch.load(pretrained_path, map_location='cuda')
        else:
            state = torch.load(pretrained_path, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        weights = state['state_dict'] if 'state_dict' in state.keys() else state

        for k, v in weights.items():
            name = k[7:] if 'module' in k and not resume else k
            new_state_dict[name] = v

        if no_strict:
            net.load_state_dict(new_state_dict, strict=False)  # ignore intermediate output
        else:
            net.load_state_dict(new_state_dict)  # optimizer has no argument `strict`

        if return_epoch_iter:
            epoch = state['epoch'] if 'epoch' in state.keys() else None
            num_iter = state['num_iter'] if 'num_iter' in state.keys() else None
            best_epe = state['best_epe'] if 'best_epe' in state.keys() else None
            best_epoch = state['best_epoch'] if 'best_epoch' in state.keys() else None
            return epoch, num_iter, best_epe, best_epoch


def resume_latest_ckpt(checkpoint_dir, net, net_name):
    ckpts = sorted(glob(checkpoint_dir + '/' + net_name + '*.pth'))

    if len(ckpts) == 0:
        raise RuntimeError('=> No checkpoint found while resuming training')

    latest_ckpt = ckpts[-1]
    print('=> Resume latest %s checkpoint: %s' % (net_name, os.path.basename(latest_ckpt)))
    epoch, num_iter, best_epe, best_epoch = load_pretrained_net(net, latest_ckpt, True, True)

    return epoch, num_iter, best_epe, best_epoch


def fix_net_parameters(net):
    for param in net.parameters():
        param.requires_grad = False


def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


def filter_specific_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return True
    return False


def filter_base_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return False
    return True


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    # fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    fmt = "[%(asctime)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger