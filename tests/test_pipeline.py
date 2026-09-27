"""
Crop Disease Detector - Automated Unit & Integration Test Suite
=============================================================
Validates data ingestion, preprocessing, feature extraction, model prediction,
and Grad-CAM explainability pipeline.
"""

import os
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import PLANTVILLAGE_CLASSES, get_disease_info, generate_benchmark_samples
from src.preprocessing import load_and_preprocess_image, IMG_SIZE
from src.features import calculate_vegetation_indices, estimate_lesion_severity, compute_environmental_disease_risk
from src.models import build_compact_cnn, build_mobilenet_classifier
from src.explainability import get_gradcam_heatmap, overlay_gradcam

def test_class_definitions():
    """Verify all 38 PlantVillage classes are defined with complete disease profiles."""
    assert len(PLANTVILLAGE_CLASSES) == 38
    assert "Tomato___Early_blight" in PLANTVILLAGE_CLASSES
    assert "Tomato___healthy" in PLANTVILLAGE_CLASSES
    
    info = get_disease_info("Tomato___Early_blight")
    assert info["crop"] == "Tomato"
    assert info["disease"] == "Early Blight"
    assert len(info["organic_treatment"]) > 10
    assert len(info["chemical_treatment"]) > 10

def test_preprocessing_pipeline():
    """Verify input image is correctly converted to normalized tensor."""
    dummy_img = Image.new("RGB", (300, 300), color=(50, 150, 50))
    batch_tensor, norm_array, pil_res = load_and_preprocess_image(dummy_img, target_size=IMG_SIZE)
    
    assert batch_tensor.shape == (1, 224, 224, 3)
    assert norm_array.shape == (224, 224, 3)
    assert batch_tensor.dtype == np.float32
    assert 0.0 <= np.max(batch_tensor) <= 1.0

def test_features_extraction():
    """Verify spectral vegetation indices and lesion severity calculation."""
    sample_array = np.full((224, 224, 3), 0.5, dtype=np.float32)
    sample_array[:, :, 1] = 0.8  # Elevated green
    
    indices = calculate_vegetation_indices(sample_array)
    assert "mean_exg" in indices
    assert "mean_gli" in indices
    assert indices["mean_exg"] > 0
    
    severity_pct, mask = estimate_lesion_severity(sample_array)
    assert 0.0 <= severity_pct <= 100.0

def test_environmental_disease_risk():
    """Verify disease triangle epidemiological risk calculation."""
    risk_high = compute_environmental_disease_risk(temp_c=24.0, humidity_pct=92.0, rainfall_mm=20.0, pathogen_type="Fungus")
    assert risk_high["risk_score"] >= 70.0
    assert "Severe" in risk_high["risk_level"] or "High" in risk_high["risk_level"]

    risk_low = compute_environmental_disease_risk(temp_c=38.0, humidity_pct=30.0, rainfall_mm=0.0, pathogen_type="Fungus")
    assert risk_low["risk_score"] < 50.0

def test_model_construction_and_inference():
    """Verify model instantiation and forward pass output shape."""
    model = build_compact_cnn(num_classes=38)
    dummy_input = np.random.uniform(0, 1, size=(1, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    
    assert preds.shape == (1, 38)
    assert np.isclose(np.sum(preds), 1.0, atol=1e-4)

def test_explainability_gradcam():
    """Verify Grad-CAM heatmap extraction and overlay blending."""
    model = build_compact_cnn(num_classes=38)
    dummy_input = np.random.uniform(0, 1, size=(1, 224, 224, 3)).astype(np.float32)
    heatmap = get_gradcam_heatmap(dummy_input, model)
    
    assert heatmap.ndim == 2
    assert 0.0 <= np.max(heatmap) <= 1.0
    
    dummy_pil = Image.new("RGB", (224, 224), (40, 120, 40))
    blended = overlay_gradcam(dummy_pil, heatmap)
    assert blended.size == (224, 224)

if __name__ == "__main__":
    print("Running integration tests...")
    test_class_definitions()
    test_preprocessing_pipeline()
    test_features_extraction()
    test_environmental_disease_risk()
    test_model_construction_and_inference()
    test_explainability_gradcam()
    print("ALL TESTS PASSED SUCCESSFULLY!")
