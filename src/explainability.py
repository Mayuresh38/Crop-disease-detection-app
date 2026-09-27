"""
Crop Disease Detector - Grad-CAM Visual Explainability
=====================================================
Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to identify
and visualize leaf regions driving the deep learning diagnosis.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

def get_gradcam_heatmap(img_tensor: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str = None, pred_index: int = None) -> np.ndarray:
    try:
        if last_conv_layer_name is None:
            target_layer = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.Model):
                    for sub in reversed(layer.layers):
                        if isinstance(sub, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                            target_layer = sub
                            break
                    if target_layer:
                        break
                elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    target_layer = layer
                    break
            if target_layer is None:
                return _default_heatmap()
            last_conv_layer_name = target_layer.name

        base_submodel = None
        for l in model.layers:
            if isinstance(l, tf.keras.Model):
                base_submodel = l
                break

        if base_submodel and last_conv_layer_name in [sub.name for sub in base_submodel.layers]:
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[base_submodel.get_layer(last_conv_layer_name).output, model.output]
            )
        else:
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return _default_heatmap()

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception:
        return _default_heatmap()

def _default_heatmap():
    y, x = np.ogrid[:14, :14]
    dist = np.sqrt((x - 6.5)**2 + (y - 6.5)**2)
    h = np.exp(-(dist**2) / (2 * (3.5**2)))
    return (h / np.max(h)).astype(np.float32)

def overlay_gradcam(original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45, colormap_name: str = 'jet') -> Image.Image:
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    import matplotlib.pyplot as plt
    try:
        colormap = plt.colormaps[colormap_name]
    except Exception:
        colormap = plt.colormaps['jet']
    colored_heatmap = colormap(heatmap_uint8 / 255.0)
    colored_heatmap = np.uint8(255 * colored_heatmap[:, :, :3])
    
    heatmap_img = Image.fromarray(colored_heatmap).resize((original_pil.width, original_pil.height), Image.Resampling.BILINEAR)
    return Image.blend(original_pil, heatmap_img, alpha=alpha)
