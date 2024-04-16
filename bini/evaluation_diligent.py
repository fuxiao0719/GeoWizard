from bilateral_normal_integration_numpy import bilateral_normal_integration
from scipy.io import loadmat
import numpy as np
import cv2
import os

obj_list = ["bear", "buddha", "cat", "cow", "goblet", "harvest", "pot1", "pot2", "reading"]
data_dir = "data/Fig7_diligent"
k = 2
made_all = dict()

for obj_name in obj_list:
    print(f"\nProcessing {obj_name} ...")
    normal_path = os.path.join(data_dir, obj_name, "normal_map.png")
    mask_path = os.path.join(data_dir, obj_name, "mask.png")
    K_path  = os.path.join(data_dir, obj_name, "K.txt")
    depth_gt_path = os.path.join("diligent_depth_GT", f"{obj_name}_gt.mat")

    normal_map = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/65535 * 2 - 1
    else:
        normal_map = normal_map/255 * 2 - 1

    try:
        mask = cv2.imread(os.path.join(mask_path), cv2.IMREAD_GRAYSCALE).astype(bool)
    except:
        mask = np.ones(normal_map.shape[:2], bool)


    K =np.loadtxt(K_path)
    depth_map_est, surface, *_ = bilateral_normal_integration(normal_map=normal_map,
                                                       normal_mask=mask,
                                                       k=k,
                                                       K=K,
                                                       max_iter=100,
                                                       tol=1e-4)

    depth_gt = loadmat(depth_gt_path)["depth_gt"]

    scale = np.nanmedian(depth_gt / depth_map_est)
    scaled_depth = depth_map_est * scale
    absolute_difference_map = np.abs(scaled_depth - depth_gt)
    made = np.nanmean(absolute_difference_map)
    made_all[obj_name] = made
    print(obj_name, f"MADE: {made:.3f}")

    # save absolute difference map using jet colormap
    absolute_difference_map = absolute_difference_map / 5  # error > 5 mm is clipped to 5 mm
    absolute_difference_map = np.clip(absolute_difference_map, 0, 1)
    absolute_difference_map = (absolute_difference_map * 255).astype(np.uint8)
    absolute_difference_map = cv2.applyColorMap(absolute_difference_map, cv2.COLORMAP_JET)
    absolute_difference_map[~mask] = 255
    cv2.imwrite(os.path.join(data_dir, obj_name, "absolute_difference_map.png"), absolute_difference_map)
    print(f"Saved {os.path.join(data_dir, obj_name, 'absolute_difference_map.png')}")

from pprint import pprint
pprint(made_all)





