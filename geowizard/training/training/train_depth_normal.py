# A reimplemented version in public environments by Xiao Fu and Mu Hu

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.parameter import Parameter

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import logging
import tqdm
import copy

import sys
sys.path.append("..")

from accelerate import Accelerator
import transformers
import numpy as np
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
import shutil

import diffusers
from diffusers import DiffusionPipeline, DDPMScheduler, DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import accelerate

import cv2
from utils.de_normalized import align_scale_shift
from utils.depth2normal import *

from utils.dataset_configuration import prepare_dataset, depth_scale_shift_normalization,  resize_max_res_tensor

from PIL import Image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="GeoWizard")
    
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    # parser.add_argument(
    #     "--input_rgb_path",
    #     type=str,
    #     required=True,
    #     help="Path to the input image.",
    # )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/",
        required=True,
        help="The Root Dataset Path.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--recom_resolution",
        type=int,
        default=768,
        help=(
            "The resolution for resizeing the input images and the depth/disparity to make full use of the pre-trained model from \
                from the stable diffusion vae, for common cases, do not change this parameter"
        ),
    )
    #TODO : Data Augmentation
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=70)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--class_embedding_lr_mult",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    # using EMA for improving the generalization
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    # dataloaderes
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # how many steps csave a checkpoints
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # using xformers for efficient training 
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    # noise offset?::: #TODO HERE
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    
    # validations every 5 Epochs
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )      
    # get the local rank
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def pyramid_noise_like(x, timesteps, discount=0.9):
    b, c, w_ori, h_ori = x.shape 
    u = nn.Upsample(size=(w_ori, h_ori), mode='bilinear')
    noise = torch.randn_like(x)
    scale = 1.5
    for i in range(10):
        r = np.random.random()*scale + scale # Rather than always going 2x, 
        w, h = max(1, int(w_ori/(r**i))), max(1, int(h_ori/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * (timesteps[...,None,None,None]/1000) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance
    
def main():

    ''' ------------------------Configs Preparation----------------------------'''
    # give the args parsers
    args = parse_args()
    # save  the tensorboard log files
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # tell the gradient_accumulation_steps, mix precison, and tensorboard
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True) # only the main process show the logs

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Doing I/O at the main proecss
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    ''' ------------------------Non-NN Modules Definition----------------------------'''
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    sd_image_variations_diffusers_path = 'lambdalabs/sd-image-variations-diffusers'
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_image_variations_diffusers_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(sd_image_variations_diffusers_path, subfolder="feature_extractor")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                    in_channels=8, sample_size=96,
                                                    class_embed_type = 'projection',
                                                    projection_class_embeddings_input_dim = 10,
                                                    cross_attention_dim = 768,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)

    # change the input layer of unet
    sd2_conv_in = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").conv_in
    unet.conv_in.weight = Parameter(sd2_conv_in.weight.repeat(1,2,1,1) / 2)
    logger.info("loading the unet input layer from {}".format(args.pretrained_model_name_or_path), main_process_only=True)

    # using EMA
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    # Freeze vae and set unet to trainable.
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.train() # only make the unet-trainable        

    # using xformers for efficient attentions.
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            logger.info("use xformers to speed up", main_process_only=True)

        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model
                
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # using checkpoint for saving the memories
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # how many cards did we use: accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params, params_class_embedding = [], []
    for name, param in unet.named_parameters():
        if 'class_embedding' in name:
            params_class_embedding.append(param)
        else:
            params.append(param)

    # optimizer settings
    optimizer = optimizer_cls(
        [
            {"params": params, "lr": args.learning_rate},
            {"params": params_class_embedding, "lr": args.learning_rate * args.class_embedding_lr_mult}
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # get the training dataset
    with accelerator.main_process_first():
        train_loader, dataset_config_dict = prepare_dataset(data_dir=args.dataset_path,
                                                                    batch_size=args.train_batch_size,
                                                                    test_batch=1,
                                                                    datathread=args.dataloader_num_workers,
                                                                    logger=logger)

    # because the optimizer not optimized every time, so we need to calculate how many steps it optimizes,
    # it is usually optimized by 
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)

    # We need to initialize the trackers we use, and also store our configuration.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
    
    # Here is the DDP training: actually is 4
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    # using the epochs to training the model
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train() 
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):
                
                # convert the images and the depths into lantent space.
                image_data = batch['rgb'].clip(-1., 1.)
                image_data_resized = resize_max_res_tensor(image_data, mode='rgb')

                device = image_data.device

                imgs_in_proc = TF.resize((image_data_resized +1)/2, 
                    (feature_extractor.crop_size['height'], feature_extractor.crop_size['width']), 
                    interpolation=InterpolationMode.BICUBIC, 
                    antialias=True
                )
                # do the normalization in float32 to preserve precision
                imgs_in_proc = ((imgs_in_proc.float() - clip_image_mean) / clip_image_std).to(weight_dtype)        
                imgs_embed= image_encoder(imgs_in_proc).image_embeds.unsqueeze(1).to(weight_dtype)

                depth = batch['depth']
                depth_stacked = depth.repeat(1,3,1,1)
                depth_resized = resize_max_res_tensor(depth_stacked, mode='depth') 
                depth_resized_normalized = depth_scale_shift_normalization(depth_resized)

                normal = batch['normal'].clip(-1., 1.)
                normal_resized = resize_max_res_tensor(normal, mode='normal')

                # encode latents
                h_batch = vae.encoder(torch.cat((image_data_resized, depth_resized_normalized, normal_resized), dim=0).to(weight_dtype))
                moments_batch = vae.quant_conv(h_batch)
                mean_batch, logvar_batch = torch.chunk(moments_batch, 2, dim=1)
                batch_latents = mean_batch * vae.config.scaling_factor
                rgb_latents, depth_latents, normal_latents = torch.chunk(batch_latents, 3, dim=0)
                geo_latents = torch.cat((depth_latents, normal_latents), dim=0)

                # here is the setting batch size, in our settings, it can be 1.0
                bsz = rgb_latents.shape[0]
            
                # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denosing.
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=depth_latents.device).repeat(2)
                timesteps = timesteps.long()

                # Sample noise that we'll add to the latents
                noise = pyramid_noise_like(geo_latents, timesteps) # create multi-res. noise
                
                # add noise to the depth lantents
                noisy_geo_latents = noise_scheduler.add_noise(geo_latents, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(geo_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                batch_imgs_embed = imgs_embed.repeat((2, 1, 1))  # [B*2, 1, 768]

                # hybrid hierarchical switcher 
                geo_class = torch.tensor([[0, 1], [1, 0]], dtype=weight_dtype, device=device)
                geo_embedding = torch.cat([torch.sin(geo_class), torch.cos(geo_class)], dim=-1).repeat_interleave(bsz, 0)

                domain_class = batch['domain'].to(weight_dtype)
                domain_embedding = torch.cat([torch.sin(domain_class), torch.cos(domain_class)], dim=-1).repeat(2,1)

                class_embedding = torch.cat((geo_embedding, domain_embedding), dim=-1)

                # predict the noise residual and compute the loss.
                unet_input = torch.cat((rgb_latents.repeat(2,1,1,1), noisy_geo_latents), dim=1)

                noise_pred = unet(unet_input, 
                                timesteps, 
                                encoder_hidden_states=batch_imgs_embed,
                                class_labels=class_embedding).sample  # [B, 4, h, w]
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # saving the checkpoints
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Stop training
            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            # validation each epoch by calculate the epe and the visualization depth
            if args.use_ema:    
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
    
            # validation inference here            
            # log_validation(
            #     vae=vae,
            #     image_encoder=image_encoder,
            #     feature_extractor=feature_extractor,
            #     unet=unet,
            #     args=args,
            #     accelerator=accelerator,
            #     weight_dtype=weight_dtype,
            #     scheduler=noise_scheduler,
            #     epoch=epoch,
            #     input_image_path=args.input_rgb_path,  
            # )
            
            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())
                
    # Create the pipeline for training and savet
    accelerator.wait_for_everyone()
    accelerator.end_training()
    


if __name__=="__main__":
    main()
