# GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image
### [Project Page](https://fuxiao0719.github.io/projects/geowizard/) | [Paper](https://arxiv.org/abs/2403.12013) | [Hugging Face](https://huggingface.co/spaces/lemonaddie/geowizard)
<br/>

> GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image
                                                                 
> [Xiao Fu*](http://fuxiao0719.github.io/), [Wei Yin*](https://yvanyin.net/), [Mu Hu*](https://github.com/JUGGHM), [Kaixuan Wang](https://wang-kx.github.io/), [Yuexin Ma](https://yuexinma.me/), [Ping Tan](https://ece.hkust.edu.hk/pingtan), [Shaojie Shen](https://uav.hkust.edu.hk/group/), [Dahua Lin](http://dahua.site/) , [Xiaoxiao Long](https://www.xxlong.site/)
> * Equal contribution              
> ECCV, 2024

<!-- ![demo_vid](assets/teaser.png) -->
Point Cloud Rendering Using Depth
![monocular](assets/depth2pcd.gif)
Image Relighting Using Normal
![monocular](assets/normal_relighting.gif)
Hair-level Details
<img src=assets/hair-level_detail.png width=100% />

## News 
- `[2024/7/05]` Check out our [Metric3D v2](https://github.com/YvanYin/Metric3D), a sota depth and normal model in terms of accuracy.
- `[2024/7/02]` Paper accepted to [ECCV'24](https://eccv.ecva.net/).
- `[2024/4/16]` Release [GeoWizard V2](https://github.com/fuxiao0719/GeoWizard), a version with more robust and three-dimensional normal.
- `[2024/3/25]` Thanks to [Kijai](https://github.com/kijai) for incorporating GeoWizard into [ComfyUI Version](https://github.com/kijai/ComfyUI-Geowizard).
- `[2024/3/19]` Release [paper](https://arxiv.org/pdf/2403.12013), [project page](https://fuxiao0719.github.io/projects/geowizard/), and [code](https://github.com/fuxiao0719/GeoWizard).

## 🛠️ Setup

We test our codes under the following environment: `Ubuntu 22.04, Python 3.9.18, CUDA 11.8`.
1. Clone this repository.
```bash
git clone git@github.com:fuxiao0719/GeoWizard.git
cd GeoWizard
```
2. Install packages
```bash
conda create -n geowizard python=3.9
conda activate geowizard
pip install -r requirements.txt
cd geowizard
```

## 🤖 Usage

### Run inference for depth & normal

Place your images in a directory `input/example` (for example, where we have prepared several cases), and run the following inference. The depth and normal outputs will be stored in `output/example`.

```bash
python run_infer.py \
    --input_dir ${input path} \
    --output_dir ${output path} \
    --ensemble_size ${ensemble size} \
    --denoise_steps ${denoising steps} \
    --seed ${seed} \
    --domain ${data type}
# e.g.
python run_infer.py \
    --input_dir input/example \
    --output_dir output \
    --ensemble_size 3 \
    --denoise_steps 10 \
    --seed 0 \
    --domain "indoor"
```

Inference settings: `--domain`: Data type. Options: "indoor", "outdoor", and "object". Note that "object" is best for background-free objects, like that in objaverse. We find that "indoor" will suit in most scenarios. Default: "indoor". `--ensemble_size` and `--denoise_steps`: trade-off arguments for speed and performance, more ensembles and denoising steps to get higher accuracy. Default: 3 and 10 (For academic comparison, please set 10 and 50, respectively). 

### Run inference for depth & normal (v2)

We additionally train a v2-model with some architecture modifications (replace image CLIP with three types of text embeddings). Now it can generate more realistic and three-dimensional normal maps on some rare images (e.g., cartoon style, see below). 

```bash
python run_infer_v2.py \
    --input_dir ${input path} \
    --output_dir ${output path} \
    --ensemble_size ${ensemble size} \
    --denoise_steps ${denoising steps} \
    --seed ${seed} \
    --domain "indoor"
# e.g.
python run_infer_v2.py \
    --input_dir input/example \
    --output_dir output \
    --ensemble_size 3 \
    --denoise_steps 10 \
    --seed 0 \
    --domain "indoor"

python run_infer_v2.py \
    --input_dir input/example_object \
    --output_dir output_object \
    --ensemble_size 3 \
    --denoise_steps 10 \
    --seed 0 \
    --domain "object"
```

<img src=assets/object_20240413.jpg width=85% />

### Run inference for 3D reconstruction using BiNI algorithm

First put the generated depth & normal npy files under the folder `bini/data` along with the segmented foreground mask (mask.png. If not set, it will utilize the whole image as mask). We provide two examples for the data structure. Then run the command as follow.

```bash
cd ../bini

python bilateral_normal_integration_numpy.py \
    --path ${input path} \
    -k ${k} \
    --iter ${iterations} \
    --tol ${tol}

# e.g. (paper setting)
python bilateral_normal_integration_numpy.py --path data/test_1 -k 2 --iter 50 --tol 1e-5
```

### Training
Here we provide two training scripts `train_depth_normal.py` and `train_depth_normal_v2.py`. You need to modify the configs accordingly. We use 8GPUs for training as default, and you can switch `8gpu.yaml` to `1gpu.yaml` with fewer computing resources. We provide our dataloader format in `dataloader/mix_loader.py` and encourage you to train it on your own customized datasets.

```bash
cd training/scripts

# v1 model
sh train_depth_normal.sh

# v2 model
sh train_depth_normal_v2.sh
```

## 📚 Related Work
We also encourage readers to follow these concurrent exciting works.
- [Marigold](https://arxiv.org/abs/2312.02145): a finetuned diffusion model for estimating monocular depth.
- [Wonder3D](https://arxiv.org/abs/2310.15008): generate multi-view normal maps and color images and reconstruct high-fidelity textured mesh.
- [HyperHuman](https://arxiv.org/abs/2310.08579): a latent structural diffusion and a structure-guided refiner for high-resolution human generation.
- [GenPercept](https://arxiv.org/abs/2403.06090): a finetuned UNet for a lot of downstream image understanding tasks.
- [Metric3D v2](https://github.com/YvanYin/Metric3D): a discriminative metric depth and surface normal estimator.
- [IC-Light](https://github.com/lllyasviel/IC-Light): text-conditioned relighting model and background-conditioned relighting model.

## 🔗 Citation & License 

```bibtex
@article{fu2024geowizard,
  title={GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image},
  author={Fu, Xiao and Yin, Wei and Hu, Mu and Wang, Kaixuan and Ma, Yuexin and Tan, Ping and Shen, Shaojie and Lin, Dahua and Long, Xiaoxiao},
  journal={arXiv preprint arXiv:2403.12013},
  year={2024}
}
```

The GeoWizard project is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) License.