# Flow Matching for MNIST: Residual vs. Base U-Net

This repository implements **Flow Matching (FM)** for generating MNIST digits. The project specifically explores the performance gap between a standard Convolutional U-Net and a Residual U-Net architecture when learning the continuous velocity fields required for probability path reconstruction.

## 🚀 Overview

Flow Matching is a simulation-free approach to training Continuous Normalizing Flows. Unlike traditional Diffusion models that rely on Gaussian noise schedules, Flow Matching learns to predict the **velocity vector field** $v_t(x)$ that pushes a simple base distribution (noise) toward a target distribution (MNIST digits).

### Key Features
* **Dual Architectures:** Comparative implementation of a standard U-Net and a Residual U-Net with GroupNorm and GELU activations.
* **Flow Visualization:** Custom `matplotlib` engine to visualize velocity fields using block-averaged gradients.
* **Experiment Tracking:** Full integration with **MLflow** for hyperparameter logging and artifact (snapshot) management.
* **Cinematic Rendering:** Tools to generate comparison GIFs with linear, quadratic, or logarithmic temporal pacing.

---

## 📊 Results: Residual vs. Base U-Net

The following visualization demonstrates the generation process over $t \in [0, 1]$. 

* **Row 1 (Residual U-Net):** Produces sharp, structurally sound digits with smooth velocity transitions.
* **Row 2 (Base U-Net):** Exhibits significant artifacts, struggling to resolve the fine topology of the digits.

![Model Comparison](models_comparison.gif)

---

## 🏗️ Architectures

### 1. Residual U-Net (Recommended)
Uses `ResidualBlock` modules that facilitate better gradient flow. This is crucial for Flow Matching as the network must learn precise velocities at every time step $t$.
* **Time Embedding:** MLP embeddings injected into every block.
* **Normalization:** GroupNorm for stable training with small batch sizes.

### 2. Base U-Net
A standard encoder-decoder structure. While capable of learning the general density, it often fails to capture the high-frequency details of the MNIST distribution, leading to the artifacts seen in the visualization above.

---

## 🛠️ Usage

### Installation
```bash
pip install torch torchvision mlflow matplotlib numpy
