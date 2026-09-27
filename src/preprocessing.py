"""
Crop Disease Detector - Image Preprocessing & Augmentation Pipeline
===================================================================
Handles tensor conversion, normalization, resizing, and data augmentation.
"""

import io
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

def load_and_preprocess_image(image_input, target_size=IMG_SIZE):
    if isinstance(image_input, (str, bytes, io.BytesIO)):
        if isinstance(image_input, str):
            pil_img = Image.open(image_input)
        else:
            pil_img = Image.open(io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
        
    resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    batch_tensor = np.expand_dims(img_array, axis=0)
    
    return batch_tensor, img_array, pil_img

def build_data_augmentation_layers():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="crop_data_augmentation")

def denormalize_image_array(img_array):
    return np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
