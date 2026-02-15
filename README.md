# Conditional Flow Matching: Semantic Control in Generative ODEs

This branch extends the previous Flow Matching implementation by introducing **Conditional Generation**. While the unconditional model learned the general distribution of MNIST, this version allows for precise control over the output by conditioning the learned velocity field on target digit labels.

## 🔬 The Evolution: Unconditional to Conditional

The core task of Flow Matching remains: learning a vector field $v_t(x)$ that transports noise to data. However, the model now accepts a label $y$, effectively learning a class-conditional probability path.

### Key Architectural Updates:
* **Label Embedding:** Introduced an `nn.Embedding` layer that maps digit classes (0–9) into a high-dimensional semantic space.
* **Feature Fusion:** The time embedding and label embedding are concatenated and injected into the `ResidualBlocks` via a projection layer.
* **Deterministic Noise Testing:** By starting multiple conditions from the *exact same* latent noise realization, we can isolate the effect of the conditioning signal on the flow geometry.

---

## 📺 Visualizing Controlled Flows

The GIF below demonstrates the "steering" capability of the models. 

![Conditional Evolution](architecture_comparison_cond.gif)

### Grid Configuration:
* **Columns (1 → 5):** Each column represents a different digit condition, but **all start from the same initial noise**.
* **Rows:** Top row shows the **Base UNet**; Bottom row shows the **Residual UNet**.
* **Overlays:** Quiver plots visualize the gradient of the predicted velocity map, showing how the "push" changes based on the target digit.

---

## 📊 Observations & Insights

### 1. The "Semantic Lift"
One of the most interesting findings was the effect of conditioning on the **Base UNet**. While the base model still produces more artifacts than the Residual version, the introduction of labels significantly improves its performance. Semantic guidance helps even a weaker backbone form recognizable structures.

### 2. Stability vs. Guidance
The **Residual UNet** remains the superior architecture, producing realistic and stable digits. However, the experiment proves that conditioning doesn't just steer the generation, it reshapes the entire transport dynamics. The geometry of the learned flow becomes label-aware, making the ODE integration more efficient.

---

## 🔜 Next Steps

* **Architectural Refinement:** Implementation of Multi-Head Self-Attention blocks to capture global dependencies.
* **Dataset Transition:** Moving toward the **NYU-Depth V2** dataset to explore **Image-to-Image conditioning**, where the model must reconstruct depth fields from RGB inputs.
