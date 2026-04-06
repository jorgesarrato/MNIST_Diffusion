# Monocular Depth Estimation via Flow Matching

A PyTorch implementation of conditional flow matching for monocular depth estimation, trained on the SUN RGB-D dataset. The model learns a vector field that transports Gaussian noise to a depth map, conditioned on an RGB image. It utilizes a UNet backbone with AdaGroupNorm conditioning and multi-scale cross-attention powered by a pretrained **DINOv2** encoder.

---

## Quantitative Results

The following evaluation metrics are computed on a 50-image subset of the SUN RGB-D test set using metric depth in meters. The median over 100 stochastic inference passes is used as the deterministic depth prediction for computing standard error and accuracy metrics. We also report the Calibration Area Error (AE) to measure the reliability of the probabilistic uncertainty map.

| Model Variant | Abs Rel ↓ | RMSE ↓ | RMSE log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ | Calib. Area Error ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DINOv2 (Base) | 0.224 | 0.655 | 0.263 | 0.671 | 0.886 | 0.956 | **0.137** |
| DINOv2 + Attn | 0.228 | 0.652 | 0.257 | 0.677 | 0.888 | 0.964 | **0.137** |
| **DINOv2 + Attn + Cross-Attn** | **0.174** | **0.531** | **0.212** | **0.763** | **0.925** | **0.978** | 0.166 |
| DINOv2 + Attn + Cross-Attn + FullLoss | 0.193 | 0.559 | 0.222 | 0.754 | 0.916 | 0.969 | 0.159 |

---

## Qualitative Results & Visualizations

Because Flow Matching yields a probabilistic posterior, we run multiple inference passes to extract both a deterministic **Median Depth** and a pixel-wise **Uncertainty Map (Standard Deviation)**.

### Model Comparison Grid
*Ground Truth and Median Predictions use a shared dynamic colormap locked to the metric range of the true depth. Uncertainty uses a fixed 0.0m–1.5m scale.*

![Model Comparison Grid](comparisons/grid_sample_1.png)

### Flow Evolution
*Logarithmic pacing of the depth formation process (more frames near `t=1` where structure solidifies).*

![Combined Evolution](comparisons/combined_evolution_1.gif)

### Uncertainty Calibration
*Reliability diagrams showing expected vs. observed quantiles across valid pixels.*

![Calibration Curves](comparisons/all_calibration_curves.png)

---

## Repository Structure

```
├── models.py             # UNet_FM, ResNet/ViT Encoders, MHA blocks (Rel Bias / SinCos)
├── datasets.py           # sun_depth_dataset (scale-normalized), nyu_depth_dataset
├── readers.py            # SUN RGB-D and NYU-v2 loaders, dataset manifest caching
├── losses.py             # FlowMatchingLoss (v-loss, x1-loss, gradient, edge-aware, SI)
├── train_FM.py           # Training loop, EMA, multi-metric DDP sync, evaluation
├── run_experiment.py     # Entry point: CLI overrides, DDP init, MLflow logging
├── compare_dinov2.py     # Multi-model eval: metrics, grid plotting, calibration, GIFs
├── evolve.py             # ODE integration (Heun), full-resolution sliding-window inference
├── visualize_final.py    # Standard single-model visualizations
├── config.py             # Base hyperparameters, paths, and CLI argparse logic
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
y ──► DINOv2  ──► global_emb ──► AdaGN at every stage                     │
└──► scale_feats ──► cross-attn at each Up stage ◄──────────┘
│
Up0 ──► Up1 ──► ... ──► output (1ch)
```

**Positional Embeddings & Attention:**
* **Self-Attention:** Applied at the bottleneck, utilizing **Relative Positional Bias** to capture sharp local geometric relationships and edges.
* **Multi-Scale Cross-Attention:** Each up-stage (except the finest) cross-attends to a spatially matching encoder feature. Because the sequence lengths of the UNet and the DINOv2 encoder often do not match physically, cross-attention utilizes **SinCos** absolute positional embeddings to align the grids safely without broadcasting errors.

### Conditioning Encoders
* **`ViT_Encoder` (DINOv2 - Default):** Uses state-of-the-art self-supervised Vision Transformer tokens. Patch tokens are projected to spatial features, while the CLS token produces the global embedding for AdaGN.
* **`ResNet_Encoder`:** A pretrained ResNet18 split into four stages. Each stage produces a spatial feature map projected to match the decoder's channel count.

---

## Dataset — SUN RGB-D & Scale Normalization

[SUN RGB-D](https://rgbd.cs.princeton.edu/) contains indoor RGB-D images collected from four physically different sensors (Kinect v1/v2, RealSense, Asus). Because these cameras have different focal lengths and native resolutions, feeding raw crops causes severe **scale ambiguity** for monocular depth estimation.

### Preprocessing pipeline

To fix this, the dataset normalizes the physical scale *before* cropping:

1. **Bit-shift decode:** `(v >> 3) | (v << 13)` → depth in metres.
2. **Border removal:** Crop to the bounding box of valid (`> 0`) depth pixels.
3. **Scale Normalization:** Resize the shortest edge to `cache_size=256` (maintaining the rectangular aspect ratio). This normalizes the pixel-to-meter scale across all cameras.
4. **Training Crop:** Cut a `168x168` window out of the normalized rectangle using `RandomCrop(168)`.
5. **Inference / Visualization:** The full, uncropped `256 x (scaled width)` rectangle is passed directly to the sliding window algorithm.

In DDP runs, rank 0 builds and serializes the three dataset splits to `dataset_cache.pt`; all other ranks load from that file after a `dist.barrier()`.

---

## Inference

### Sliding Window Inference (`evolve.py`)
Full-resolution predictions dynamically operate on the natively scaled rectangles (e.g., `256 x 341`). Overlapping `168x168` patches are integrated via a **Heun (trapezoidal)** ODE solver and stitched together using a triangular blend window (`linspace(0.1→1.0) ⊗ linspace(1.0→0.1)`) to suppress seam artifacts.

### Classifier-Free Guidance
During training, `cond_drop_prob` fraction of samples have their conditioning zeroed out, teaching the model an unconditional branch. At inference, guided velocity is:

```
v_guided = v_uncond + w · (v_cond − v_uncond)
```

---

## Loss Function (`losses.py`)

`FlowMatchingLoss` is a modular, weighted sum of multiple objectives masked on valid depth pixels:

| Term | Flag | Description |
|---|---|---|
| `SmoothL1(β=0.1)` on `v` | Base | Base Maximum Likelihood flow-matching loss on the velocity field. |
| `SmoothL1` on `x_1` | `x1_weight` | Reconstruction loss on the inferred clear depth map. |
| Gradient loss | `grad_weight` | Penalizes blurry edges based on spatial depth gradients. |
| Edge-aware loss | `edge_weight` | Gradient loss weighted by `exp(-∥∇RGB∥)` to emphasize depth discontinuities at physical object edges. |
| Scale-invariant loss | `si_weight` | Log-space variance term to help contextualize global depth. |

*All specific sub-metrics are synced across DDP ranks and logged to MLflow independently.*

---

## Training (`train_FM.py`)

**EMA**: an `AveragedModel` with `get_ema_multi_avg_fn(decay)` shadows the live model throughout training. Validation and checkpointing always use the EMA model. 

**Mixed precision**: `torch.amp.GradScaler` + `autocast('cuda')` throughout. Gradient norm is clipped to 0.5 before every optimiser step.

**DDP**: Train loss is all-reduced across ranks after each epoch. Validation metrics are all-reduced after each eval pass. Checkpointing and MLflow logging run on rank 0 only.

---

## Installation & Usage

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
cp .env.example .env   # Set DATA_MYSUNRGBD_DIR, MLFLOW_DIR, etc.
```

### Single GPU:

```bash
python src/run_experiment.py
```

### Multi-GPU DDP:

```bash
torchrun --nproc_per_node=NUM_GPUS src/run_experiment.py
```

### Multi-Model Evaluation & Comparison:

```bash
torchrun --nproc_per_node=NUM_GPUS src/compare_dinov2.py
```

### Requirements:

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
imageio
```

## Acknowledgements

- [SUN RGB-D](https://rgbd.cs.princeton.edu/) — Song et al., CVPR 2015
- [Flow Matching for Generative Modelling](https://arxiv.org/abs/2210.02747) — Lipman et al., ICLR 2023
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — Oquab et al., 2023
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Ho & Salimans, 2022