"""
Crop Disease Detector - Agronomic & Vision Feature Extraction
=============================================================
Calculates vegetation indices (ExG, GLI, NDGR), necrotic lesion coverage,
and agronomic environmental disease spread risk index.
"""

import numpy as np

def calculate_vegetation_indices(img_array: np.ndarray) -> dict:
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    gli = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-6)
    
    return {
        "mean_exg": float(np.mean(exg)),
        "mean_exr": float(np.mean(exr)),
        "mean_gli": float(np.mean(gli)),
        "mean_green_intensity": float(np.mean(g)),
        "mean_red_intensity": float(np.mean(r)),
        "mean_blue_intensity": float(np.mean(b)),
    }

def estimate_lesion_severity(img_array: np.ndarray) -> tuple:
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    is_leaf = (g > 0.15) & ((r + g + b) < 2.65)
    is_lesion = is_leaf & ((r > g * 0.95) | ((r < 0.25) & (g < 0.25) & (b < 0.25)))
    
    leaf_pixels = np.sum(is_leaf)
    if leaf_pixels == 0:
        return 0.0, np.zeros_like(r, dtype=bool)
        
    lesion_pixels = np.sum(is_lesion)
    lesion_pct = (lesion_pixels / leaf_pixels) * 100.0
    lesion_pct = min(100.0, max(0.0, float(lesion_pct)))
    
    return lesion_pct, is_lesion

def compute_environmental_disease_risk(temp_c: float, humidity_pct: float, rainfall_mm: float, pathogen_type: str = "Fungus") -> dict:
    risk_score = 0.0
    factors = []
    
    if "fung" in pathogen_type.lower() or "oomycete" in pathogen_type.lower():
        if 18.0 <= temp_c <= 28.0:
            risk_score += 35.0
            factors.append("Optimal temperature window for fungal sporulation (18-28C)")
        elif 14.0 <= temp_c < 18.0 or 28.0 < temp_c <= 32.0:
            risk_score += 20.0
            factors.append("Moderate thermal condition for pathogen activity")
    else:
        if 24.0 <= temp_c <= 33.0:
            risk_score += 35.0
            factors.append("Optimal temperature for bacterial multiplication (24-33C)")
        else:
            risk_score += 15.0
            
    if humidity_pct >= 85.0:
        risk_score += 40.0
        factors.append("High relative humidity (>85%) facilitates rapid spore germination")
    elif humidity_pct >= 70.0:
        risk_score += 25.0
        factors.append("Elevated humidity (>70%) maintains leaf wetness")
    else:
        risk_score += 10.0
        
    if rainfall_mm >= 15.0:
        risk_score += 25.0
        factors.append("Heavy rainfall induces soil-to-canopy spore splash")
    elif rainfall_mm >= 5.0:
        risk_score += 15.0
        factors.append("Moderate rain provides continuous surface moisture film")
    else:
        risk_score += 5.0
        
    risk_score = min(100.0, risk_score)
    
    if risk_score >= 80.0:
        level = "Severe Outbreak Warning"
        alert_color = "#e53935"
        advice = "Immediate prophylactic fungicide/bactericide spray recommended. Minimize field transit."
    elif risk_score >= 55.0:
        level = "High Infection Risk"
        alert_color = "#fb8c00"
        advice = "High micro-climate favorability. Increase scouting frequency to 48-hour cycles; ensure field drainage."
    elif risk_score >= 35.0:
        level = "Moderate Risk"
        alert_color = "#fdd835"
        advice = "Conditions allow localized transmission. Monitor lower canopy and avoid overhead irrigation."
    else:
        level = "Low Environmental Risk"
        alert_color = "#43a047"
        advice = "Dry, sub-optimal conditions for foliar pathogens. Continue standard cultural routines."
        
    return {
        "risk_score": round(risk_score, 1),
        "risk_level": level,
        "alert_color": alert_color,
        "factors": factors,
        "action_advice": advice
    }
