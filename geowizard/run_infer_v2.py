# Adapted from Marigold ：https://github.com/prs-eth/Marigold

import argparse
import os
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import glob
import json
import cv2

import sys
from models.geowizard_v2_pipeline import DepthNormalEstimationPipeline
from utils.seed_all import seed_all
import matplotlib.pyplot as plt
from utils.depth2normal import *

from diffusers import DiffusionPipeline, DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer

if __name__=="__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run MonoDepthNormal Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='lemonaddie/geowizard',
        help="pretrained model path from hugging face or local dir",
    )    
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory."
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default='object',
        help="domain prediction",
    )   

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    
    args = parser.parse_args()
    
    checkpoint_path = args.pretrained_model_path
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size>15:
        logging.warning("long ensemble steps, low speed..")
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    domain = args.domain

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    
    if batch_size==0:
        batch_size = 1  # set default batchsize
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # Output directories
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    output_dir_normal_npy = os.path.join(output_dir, "normal_npy")
    output_dir_normal_color = os.path.join(output_dir, "normal_colored")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_normal_npy, exist_ok=True)
    os.makedirs(output_dir_normal_color, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    input_dir = args.input_dir
    test_files = sorted(os.listdir(input_dir))
    n_images = len(test_files)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # declare a pipeline
    stable_diffusion_repo_path = "stabilityai/stable-diffusion-2"
    vae = AutoencoderKL.from_pretrained(stable_diffusion_repo_path, subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_repo_path, subfolder='text_encoder')
    scheduler = DDIMScheduler.from_pretrained(stable_diffusion_repo_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_repo_path, subfolder='tokenizer')
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder='unet_v2')
                
    pipe = DepthNormalEstimationPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                unet=unet,
                                scheduler=scheduler)

    logging.info("loading pipeline whole successfully.")
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for test_file in tqdm(test_files, desc="Estimating Depth & Normal", leave=True):
            rgb_path = os.path.join(input_dir, test_file)

            # Read input image
            input_image = Image.open(rgb_path)

            # predict the depth & normal here
            pipe_out = pipe(input_image,
                denoising_steps = denoise_steps,
                ensemble_size= ensemble_size,
                processing_res = processing_res,
                match_input_res = match_input_res,
                domain = domain,
                color_map = color_map,
                show_progress_bar = True,
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored
            normal_pred: np.ndarray = pipe_out.normal_np
            normal_colored: Image.Image = pipe_out.normal_colored

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)

            normal_npy_save_path = os.path.join(output_dir_normal_npy, f"{pred_name_base}.npy")
            if os.path.exists(normal_npy_save_path):
                logging.warning(f"Existing file: '{normal_npy_save_path}' will be overwritten")
            np.save(normal_npy_save_path, normal_pred)

            # Colorize
            depth_colored_save_path = os.path.join(output_dir_color, f"{pred_name_base}_colored.png")
            if os.path.exists(depth_colored_save_path):
                logging.warning(
                    f"Existing file: '{depth_colored_save_path}' will be overwritten"
                )
            depth_colored.save(depth_colored_save_path)

            normal_colored_save_path = os.path.join(output_dir_normal_color, f"{pred_name_base}_colored.png")
            if os.path.exists(normal_colored_save_path):
                logging.warning(
                    f"Existing file: '{normal_colored_save_path}' will be overwritten"
                )
            normal_colored.save(normal_colored_save_path)
