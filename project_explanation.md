# Architectural & Technical Explanation: Crop Disease Detector

This document details the engineering decisions, biophysical equations, deep learning paradigms, and explainability frameworks incorporated into the Crop Disease Diagnostic platform.

---

## 1. Transfer Learning with MobileNetV2

Crop foliage classification presents fine-grained visual distinctions where lesions differ subtly across fungal species (e.g., concentric bullseye rings in Early Blight vs. water-soaked margins in Late Blight).

- **Inverted Residuals & Linear Bottlenecks:** MobileNetV2 leverages depthwise separable convolutions that dramatically reduce parameter count (~3.4M params) while preserving representation capacity.
- **Dense Custom Head:**
  - `GlobalAveragePooling2D()` flattens spatial activations into a 1280-dimensional feature vector.
  - `BatchNormalization()` normalizes feature scale before dense projections.
  - `Dense(256, activation='relu')` with `Dropout(0.4)` mitigates co-adaptation.
  - `Dense(38, activation='softmax')` outputs multi-class probability distributions.

---

## 2. Biophysical & Spectral Feature Engineering

To complement black-box neural networks, biophysical spectral telemetry is derived:

### Excess Green Index (ExG)
$$\text{ExG} = 2G - R - B$$
Isolates green photosynthetic foliage from brown necrotic lesions and neutral soil backgrounds.

### Green Leaf Index (GLI)
$$\text{GLI} = \frac{2G - R - B}{2G + R + B + \epsilon}$$
Provides a normalized bounded metric `[-1.0, +1.0]` of canopy vitality.

---

## 3. Grad-CAM (Gradient-Weighted Class Activation Mapping)

Ensures clinical and agronomic trustworthiness by answering: *Which pixels caused the model to classify this leaf as Late Blight?*

1. Gradients of the predicted class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer are computed via `tf.GradientTape()`.
2. Spatial pooling averages gradient importance:
$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. Weighted linear combination is rectified through ReLU:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
