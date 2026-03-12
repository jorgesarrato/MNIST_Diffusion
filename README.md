# Monocular Depth Estimation via Flow Matching

A PyTorch implementation of conditional flow matching for monocular depth estimation, trained on SUN RGB-D. The model learns a vector field that transports Gaussian noise to a depth map, conditioned on an RGB image, using a UNet backbone with AdaGroupNorm conditioning and optional multi-scale cross-attention from a pretrained ResNet18 encoder.

---

## Current Results

Next I show depth reconstruction GIFs for 4 random samples from the test set. These correspond to training with a resnet-18 backbone, multi-scale cross-attention
capped to the bottleneck and smallest upsampling blocks of the Residual U-Net, self-attention at the bottleneck, and a relatively long 500 epoch training run
using OneCycleLR. Training is done on random 128x128 scaled crops taken from 256x256 scaled-down central patches of the images in the dataset.

For each test sample I show 3 reconstructions using different scales of classifier-free guidance: 1.0 (purely conditioned reconstruction), 1.5 and 2.0.
First two samples correspond to "easy" images that are reasonably well reconstructed, whilst the two next samples present more challenging rooms full
of sub-structure that is currently not well recovered.

![sample 0 cfg 1.0](evolution_test_log_0_1.0.gif)
![sample 0 cfg 1.5](evolution_test_log_0_1.5.gif)
![sample 0 cfg 2.0](evolution_test_log_0_2.0.gif)

![sample 3 cfg 1.0](evolution_test_log_3_1.0.gif)
![sample 3 cfg 1.5](evolution_test_log_3_1.5.gif)
![sample 3 cfg 2.0](evolution_test_log_3_2.0.gif)

![sample 2 cfg 1.0](evolution_test_log_2_1.0.gif)
![sample 2 cfg 1.5](evolution_test_log_2_1.5.gif)
![sample 2 cfg 2.0](evolution_test_log_2_2.0.gif)

![sample 4 cfg 1.0](evolution_test_log_4_1.0.gif)
![sample 4 cfg 1.5](evolution_test_log_4_1.5.gif)
![sample 4 cfg 2.0](evolution_test_log_4_2.0.gif)

---

## Repository Structure

```
├── models.py           # UNet_FM, ResNet_Encoder, Image_Encoder, attention blocks
├── datasets.py         # sun_depth_dataset, nyu_depth_dataset, border removal, caching
├── readers.py          # SUN RGB-D and NYU-v2 loaders, dataset manifest caching
├── losses.py           # FlowMatchingLoss, ScaleInvariantLoss, gradient & edge-aware losses
├── train_FM.py         # Training loop, EMA, DDP metric sync, evaluation
├── run_experiment.py   # Entry point: DDP init, dataset build, MLflow, animation export
├── evolve.py           # ODE integration (Heun), single-patch and sliding-window inference
├── visualize.py        # Flow animation, depth evolution GIFs, multi-model comparison
├── config.py           # All hyperparameters and paths
└── utils/
    ├── model_parser.py
    ├── opt_parser.py
    └── scheduler_parser.py
```

---

## Architecture

### UNet Flow Matching Model (`UNet_FM`)

The denoiser is a UNet that takes a noisy depth map `x_t`, a scalar timestep `t`, and an RGB conditioning image `y`, and predicts the velocity field `v(x_t, t, y)`.

```
x_t  (noisy depth, 1ch)  ──┐
                            ├─ concat ──► Down0 ──► Down1 ──► ... ──► Bottleneck
y    (RGB, 3ch)          ──┘                                              │
                                                                     self-attn
y ──► Encoder ──► global_emb ──► AdaGN at every stage                    │
              └──► scale_feats ──► cross-attn at each Up stage ◄──────────┘
                                         │
                                        Up0 ──► Up1 ──► ... ──► output (1ch)
```

**Down / Up stages** use `ResidualBlock` with two-layer AdaGroupNorm conditioning (`t_emb ⊕ y_global`). Gradients through residual blocks and up-block skip connections use `torch.utils.checkpoint` during training to reduce activation memory.

**Bottleneck** applies `ResidualBlock → (optional) self-attention → ResidualBlock`. Self-attention uses a standard QKV formulation with zero-initialised output projection.

**Multi-scale cross-attention**: each up-stage (except the finest) cross-attends to a spatially matching encoder feature. Queries come from the decoder feature map; keys and values come from the encoder feature projected to the same channel count. 2D sinusoidal positional embeddings are added to both, and are cached per `(h, w)` shape to avoid recomputation. The finest up-stage skips cross-attention but is linked to the encoder graph via a `0 * enc_feat.sum()` term to keep DDP gradient synchronisation correct.

**Conditioning modes** (`cond_type`):
- `"concat"` — RGB image is channel-concatenated with `x_t` before the first conv.
- `"simple"` — conditioning enters only via AdaGN (global embedding only).

### Conditioning Encoders

**`ResNet_Encoder`** (recommended): splits a pretrained ResNet18 into four named stages (`stem`, `layer1`, `layer2`, `layer3`). The stem is frozen. Each stage produces a spatial feature map; a per-scale `Conv1×1 → GroupNorm → GELU` head projects it to exactly the decoder channel count at that depth, ensuring zero channel mismatch at every cross-attention block. The global conditioning vector is pooled from `layer3` (256ch).

| ResNet stage | Channels | Stride | Decoder up-stage |
|---|---|---|---|
| layer3 | 256 → `filters[-2]` | /16 | up-stage 0 (coarsest) |
| layer2 | 128 → `filters[-3]` | /8  | up-stage 1 |
| layer1 |  64 → `filters[-4]` | /4  | up-stage 2 |
| stem   |  64 → `filters[0]`  | /4  | (finest, unused for cross-attn) |

**`Image_Encoder`** (lightweight alternative): a small strided-conv encoder with the same `ResidualBlock` as the UNet. Also supports multi-scale feature extraction via the same `decoder_channels` API.

**`ViT_Encoder`** (experimental): `vit_small_patch16_224` from `timm`, with the first four blocks frozen. Patch tokens are projected to spatial features; the CLS token produces the global embedding.

---

## Dataset — SUN RGB-D

[SUN RGB-D](https://rgbd.cs.princeton.edu/) contains ~10k indoor RGB-D images from four sensor types (Kinect v1/v2, Intel RealSense, Asus Xtion). Depth values are decoded from the raw 16-bit PNG format using the bitshift correction `(v >> 3) | (v << 13)` and converted to metres. Values above 10 m are zeroed out.

### Preprocessing pipeline

```
Raw image + raw 16-bit depth PNG
         │
         ▼
 1. Bit-shift decode  →  depth in metres
 2. Border removal    →  crop to bounding box of valid (> 0) depth pixels
 3. Square crop       →  largest square from the cropped region (via Resize + CenterCrop)
 4. Cache resize      →  resize shorter side to cache_size, then CenterCrop to cache_size×cache_size
         │
         │  stored in RAM as float32 tensors, normalised to [-1, 1]  (depth clipped to [0.7, 10.0] m)
         ▼
 5. Training crop     →  RandomResizedCrop(crop_size, scale=(0.16, 1.0), ratio=1.0)
                          + RandomHorizontalFlip
 6. GPU augmentation  →  ColorJitter, RandomGrayscale, GaussianBlur
                          (applied on-the-fly inside the training loop)
```

Both `cache_size` and `crop_size` (`side_pixels`) are set in `config.py`. The dataset manifest (list of RGB/depth file paths) is cached to `dataset_manifest.json` on first run to avoid repeated directory scans.

In DDP runs, rank 0 builds and serialises the three dataset splits to `dataset_cache.pt`; all other ranks load from that file after a `dist.barrier()`.

---

## Sliding Window Inference (`evolve.py`)

Because the model is trained on `crop_size` patches, full-resolution predictions use a sliding window over the cached `cache_size` image:

1. Pad the image so its dimensions are evenly divisible by the stride.
2. Extract all `crop_size × crop_size` patches at the given stride.
3. Process patches in mini-batches of size `max_batch_size` to bound GPU memory.
4. Stitch predictions back using a triangular blend window (`linspace(0.1→1.0) ⊗ linspace(1.0→0.1)`) to suppress seam artefacts.
5. Crop padding back off to recover the original spatial extent.

The ODE is integrated with a **Heun (trapezoidal) method**: Euler on the last step, Heun on all others. This is applied patch-by-patch in a full-image loop so the velocity field at each timestep is spatially coherent across the stitched result.

---

## Classifier-Free Guidance

During training, `cond_drop_prob` fraction of samples have their conditioning zeroed out via `drop_mask`, teaching the model an unconditional branch. At inference, guided velocity is:

```
v_guided = v_uncond + w · (v_cond − v_uncond)
```

Multiple guidance scales can be swept in a single run via `guidance_scale` in `training_config` (e.g. `[1.0, 1.5, 2.0]`). Each scale produces a separate animation artifact logged to MLflow.

---

## Loss Function (`losses.py`)

`FlowMatchingLoss` is a weighted sum of up to four terms, all computed per-sample and masked on valid depth pixels (`depth > -0.99` in normalised space):

| Term | Flag | Description |
|---|---|---|
| `SmoothL1(β=0.05)` on `v_pred` vs `v` | always active | Base flow-matching loss on the velocity field |
| Gradient loss on `x1_pred` vs `x` | `grad_weight` | L1 on spatial depth gradients; penalises blurry edges |
| Edge-aware gradient loss | `edge_weight` | Gradient loss weighted by `exp(-∥∇RGB∥)` so depth discontinuities at RGB edges are emphasised |
| Scale-invariant loss (`λ=0.85`) | `si_weight` | Log-space variance term; handles scale ambiguity in metric depth |

Loss weights per sample are optionally modulated by a quadratic time weight `w(t) = t² + 0.1` (`weight_type="quad"`), upweighting harder timesteps near `t=1`.

---

## Training (`train_FM.py`)

**EMA**: an `AveragedModel` with `get_ema_multi_avg_fn(decay)` shadows the live model throughout training. Validation and checkpointing always use the EMA model. Non-parameter buffers (e.g. BatchNorm running stats) are synced from the base model before each validation pass.

**Mixed precision**: `torch.amp.GradScaler` + `autocast('cuda')` throughout. Gradient norm is clipped to 0.5 before every optimiser step.

**Schedulers**: batch-level schedulers (`OneCycleLR`, `CyclicLR`, `CosineAnnealingWarmRestarts`) are stepped every batch; epoch-level schedulers step after each validation. `ReduceLROnPlateau` uses the total validation loss.

**DDP**: train loss is all-reduced across ranks after each epoch. Validation metrics are all-reduced after each eval pass. Checkpointing and MLflow logging run on rank 0 only.

**Evaluation metrics logged per epoch**:
- `val_total` — weighted composite loss (used for early stopping and best-model saving)
- `val_l1` — masked SmoothL1 on the velocity field
- `val_grad` — gradient loss on the predicted depth
- `val_si` — scale-invariant loss on the predicted depth

---

## Visualisation (`visualize.py`)

After training, `run_experiment.py` generates full-resolution sliding-window flow animations for 5 train and 5 test samples at each configured guidance scale. Each animation shows three panels side by side:

```
[ RGB condition ] [ Ground truth depth ] [ Predicted depth + velocity quiver ]
```

Animations are saved as GIFs with **logarithmic frame pacing** (more frames near `t=1` where the depth is already formed) and logged as MLflow artifacts.

`create_multi_model_flow_animation` supports grid comparison of multiple model checkpoints across multiple samples in a single GIF.

---

## Configuration (`config.py`)

All paths are read from a `.env` file via `python-dotenv`.

| Key | Default | Description |
|---|---|---|
| `cache_size` | `128` | Square side length (px) images are resized to before caching |
| `side_pixels` | `128` | Square side length (px) of training crops and inference patches |
| `batch_size` | `32` | Per-GPU batch size |
| `val_split` | `0.1` | Fraction held out for val and test respectively |
| `filters_arr` | `[32,64,128,256]` | UNet channel progression |
| `t_emb_size` | `512` | Sinusoidal time embedding dimension |
| `label_emb_size` | `1024` | Global conditioning vector dimension |
| `encoder_type` | `"simple"` | `"resnet"` or `"simple"` |
| `cross_attn` | `True` | Enable multi-scale cross-attention |
| `use_residuals` | `True` | AdaGN residual blocks throughout the UNet |
| `attn` | `True` | Self-attention at the bottleneck |
| `cond_type` | `"simple"` | `"concat"` or `"simple"` |
| `scheduler` | `OneCycleLR` | LR scheduler |
| `pct_start` | `0.10` | Fraction of training spent warming up |
| `ema_decay` | `0.999` | EMA decay coefficient |
| `cond_drop_prob` | `0.10` | Probability of dropping conditioning (CFG training) |
| `guidance_scale` | `[1.0,1.5,2.0]` | CFG scales evaluated at inference |
| `grad_weight` | `0.0` | Weight of gradient loss term |
| `weight_type` | `"none"` | Time-weighting scheme (`"quad"` or `"none"`) |
| `time_sampling` | `"uniform"` | `"uniform"` or `"logit_normal"` |

---

## Installation & Usage

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
cp .env.example .env   # set DATA_SUNRGBD_DIR, MLFLOW_DIR, etc.
```

**Single GPU:**
```bash
python run_experiment.py
```

**Multi-GPU DDP:**
```bash
torchrun --nproc_per_node=NUM_GPUS run_experiment.py
```

**Requirements (core):**
```
torch >= 2.2
torchvision
timm
numpy
Pillow
tqdm
h5py
mlflow
scikit-learn
python-dotenv
matplotlib
```

---

## Acknowledgements

- [SUN RGB-D](https://rgbd.cs.princeton.edu/) — Song et al., CVPR 2015
- [Flow Matching for Generative Modelling](https://arxiv.org/abs/2210.02747) — Lipman et al., ICLR 2023
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Ho & Salimans, 2022
- ResNet18 backbone from `torchvision`, pretrained on ImageNet-1k