"""
Crop Disease Detector - Deep Learning Architecture
==================================================
Implements Transfer Learning classifier using MobileNetV2 with custom dense head,
supporting rapid inference and Grad-CAM interpretability.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 38
INPUT_SHAPE = (224, 224, 3)

def build_mobilenet_classifier(num_classes: int = NUM_CLASSES, input_shape: tuple = INPUT_SHAPE) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="leaf_image_input")
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dense(256, activation="relu", name="dense_features_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_features_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="PlantVillage_MobileNetV2")
    return model

def build_compact_cnn(num_classes: int = NUM_CLASSES, input_shape: tuple = INPUT_SHAPE) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape, name="leaf_image_input"),
        
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv_final"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.GlobalAveragePooling2D(name="global_avg_pool"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax", name="disease_prediction")
    ], name="Compact_Crop_CNN")
    
    return model

def load_or_initialize_model(model_path: str, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded existing model from: {model_path}")
            return model
        except Exception as e:
            print(f"Notice: Could not load saved model ({e}), initializing fresh architecture.")
            
    model = build_mobilenet_classifier(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )
    return model
