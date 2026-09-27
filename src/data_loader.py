"""
Crop Disease Detector - Data Ingestion & Disease Knowledge Base
==============================================================
Provides PlantVillage dataset class taxonomy, domain agronomic knowledge,
synthetic benchmark image generation, and metadata management.
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

DISEASE_KNOWLEDGE_BASE = {
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "pathogen": "Fungus (Alternaria solani)",
        "severity": "Moderate to High",
        "symptoms": "Dark brown circular spots with characteristic concentric rings ('bullseye' target pattern) starting on older lower foliage, surrounded by chlorotic yellow halos.",
        "organic_treatment": "Apply bio-fungicides containing Bacillus subtilis or copper octanoate spray. Strip affected lower leaves immediately and mulch soil to prevent fungal spore splash.",
        "chemical_treatment": "Foliar application of Chlorothalonil (0.2%) or Mancozeb (0.25%) at 7 to 10 day intervals during warm humid weather.",
        "prevention": "Maintain 3-year Solanaceous crop rotation, utilize drip irrigation instead of overhead sprinklers, and ensure optimal plant spacing (60 cm) for canopy airflow."
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "severity": "Critical (Rapid Crop Loss)",
        "symptoms": "Large, irregular water-soaked pale green to dark brown lesions on leaves and stems, often accompanied by white fluffy fungal mycelium on leaf undersides in high humidity.",
        "organic_treatment": "Bordeaux mixture (copper sulfate + lime) or copper hydroxide sprays applied as early protectants; remove and incinerate severely infected vines.",
        "chemical_treatment": "Systemic fungicides such as Metalaxyl + Mancozeb (Ridomil Gold) or Cymoxanil; spray proactively when night temps are 10-15C and relative humidity exceeds 90%.",
        "prevention": "Plant certified disease-free transplants, eliminate cull piles and volunteer tomato/potato plants, and monitor local late blight weather alerts."
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy Foliage",
        "pathogen": "None (Optimum Physiological Health)",
        "severity": "None",
        "symptoms": "Vibrant uniform dark-green compound leaves, robust venation, absence of lesions, spots, or chlorosis.",
        "organic_treatment": "Maintain balanced vermicompost tea or seaweed foliar spray every 2 weeks for systemic acquired resistance.",
        "chemical_treatment": "No chemical intervention needed. Avoid unnecessary prophylactic fungicide applications.",
        "prevention": "Continue regular nutrient management (NPK 4:2:4 during vegetative, 2:1:4 during fruiting), optimal soil moisture, and pest scouting."
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "pathogen": "Fungus (Alternaria solani)",
        "severity": "Moderate",
        "symptoms": "Small brown angular spots enlarging to show concentric dark rings, restricted by leaf veins; premature leaf defoliation.",
        "organic_treatment": "Spray neem seed kernel extract (5%) or copper oxychloride; remove infected leaf trash post-harvest.",
        "chemical_treatment": "Apply Azoxystrobin, Difenoconazole, or Chlorothalonil every 10-14 days.",
        "prevention": "Ensure adequate nitrogen and potassium fertilization; stressed plants are significantly more susceptible."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "severity": "Critical",
        "symptoms": "Rapidly expanding dark brown to purplish-black water-soaked lesions on leaf tips and margins; white downy mildew on undersides.",
        "organic_treatment": "Fixed copper fungicides before infection sets in; destroy infected plants immediately.",
        "chemical_treatment": "Dimethomorph, Cymoxanil, or Propamocarb hydrochloride sprays upon first symptom detection.",
        "prevention": "Plant resistant cultivars, avoid overhead watering, hill tubers well to prevent spore wash into soil."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy Foliage",
        "pathogen": "None (Healthy)",
        "severity": "None",
        "symptoms": "Uniform deep green leaves, undamaged petioles, vigorous vegetative canopy.",
        "organic_treatment": "Apply bio-fertilizer and foliar micronutrient spray to maintain vigorous canopy.",
        "chemical_treatment": "None required.",
        "prevention": "Maintain soil hilling, scouting schedule, and optimal soil drainage."
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "pathogen": "Fungus (Puccinia sorghi)",
        "severity": "Moderate",
        "symptoms": "Elongated cinnamon-brown to golden powdery pustules (uredinia) scattered on both upper and lower leaf surfaces.",
        "organic_treatment": "Sulfur-based dusts or bio-formulations of Trichoderma harzianum; early crop scouting.",
        "chemical_treatment": "Foliar triazole or strobilurin fungicides (e.g., Pyraclostrobin, Propiconazole) if pustules appear before tasseling.",
        "prevention": "Plant rust-resistant corn hybrids; early planting to avoid peak airborne urediniospore dispersal."
    },
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "pathogen": "Fungus (Venturia inaequalis)",
        "severity": "High (Fruit Quality Degradation)",
        "symptoms": "Dull olive-green to black velvety spots on leaves, turning dark brown with distinct margins; leaf puckering and early leaf drop.",
        "organic_treatment": "Liquid lime sulfur or potassium bicarbonate sprays during early green tip and petal fall stages.",
        "chemical_treatment": "Myclobutanil, Captan, or Difenoconazole applied during primary infection periods (spring rains).",
        "prevention": "Rake and shred or compost fallen autumn leaves to eliminate overwintering pseudothecia; prune trees for open sunlight."
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "disease": "Bacterial Spot",
        "pathogen": "Bacterium (Xanthomonas campestris pv. vesicatoria)",
        "severity": "High",
        "symptoms": "Small, circular, water-soaked dark spots that develop into irregular lesions with necrotic centers and yellow halos; blossom drop.",
        "organic_treatment": "Copper hydroxide mixed with Bacillus amyloliquefaciens; avoid working in fields when foliage is wet.",
        "chemical_treatment": "Copper bactericides combined with Mancozeb (to overcome copper resistance in Xanthomonas strains).",
        "prevention": "Use certified hot-water treated pathogen-free seeds; eliminate solanaceous weed hosts."
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper",
        "disease": "Healthy Foliage",
        "pathogen": "None (Healthy)",
        "severity": "None",
        "symptoms": "Glossy green leaves, strong erect branching, intact leaf lamina without chlorosis or necrosis.",
        "organic_treatment": "Foliar application of fish emulsion and seaweed extract.",
        "chemical_treatment": "None required.",
        "prevention": "Maintain soil moisture consistency to prevent blossom end rot and foliar stress."
    }
}

def get_disease_info(class_name: str) -> dict:
    """Retrieve detailed agronomic profile for a given class."""
    if class_name in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[class_name]
    
    parts = class_name.split("___")
    crop = parts[0].replace("_", " ").title()
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease.lower()
    
    return {
        "crop": crop,
        "disease": disease,
        "pathogen": "None (Healthy)" if is_healthy else f"Pathogen associated with {disease}",
        "severity": "None" if is_healthy else "Moderate",
        "symptoms": f"Standard agronomic manifestation of {disease} on {crop} foliage.",
        "organic_treatment": "Maintain balanced bio-fertilizers and organic foliar bio-fungicide sprays." if not is_healthy else "Maintain optimal soil fertility and scouting.",
        "chemical_treatment": f"Consult local extension officer for recommended labeled fungicides/bactericides against {disease}." if not is_healthy else "No chemical treatment needed.",
        "prevention": "Implement sanitary crop rotation, clean irrigation water, and resistant seed varieties."
    }

def generate_benchmark_samples(output_dir: str):
    """
    Generate realistic synthetic benchmark images of healthy and diseased leaves
    for immediate interactive verification, testing, and demonstration.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    samples_to_generate = [
        ("Tomato___Early_blight", "tomato_early_blight.jpg", "early_blight"),
        ("Tomato___Late_blight", "tomato_late_blight.jpg", "late_blight"),
        ("Tomato___healthy", "tomato_healthy.jpg", "healthy"),
        ("Potato___Early_blight", "potato_early_blight.jpg", "early_blight"),
        ("Potato___Late_blight", "potato_late_blight.jpg", "late_blight"),
        ("Corn_(maize)___Common_rust_", "corn_common_rust.jpg", "rust"),
        ("Apple___Apple_scab", "apple_scab.jpg", "scab"),
        ("Pepper,_bell___Bacterial_spot", "pepper_bacterial_spot.jpg", "bacterial_spot"),
        ("Pepper,_bell___healthy", "pepper_healthy.jpg", "healthy"),
    ]
    
    for class_name, filename, pattern in samples_to_generate:
        img_path = os.path.join(output_dir, filename)
        if os.path.exists(img_path):
            continue
            
        img = Image.new("RGB", (224, 224), (230, 235, 230))
        draw = ImageDraw.Draw(img)
        
        base_color = (45, 130, 45) if pattern != "healthy" else (34, 155, 45)
        draw.polygon([(112, 20), (190, 85), (170, 185), (112, 210), (54, 185), (34, 85)], fill=base_color)
        
        draw.line([(112, 20), (112, 210)], fill=(80, 175, 80), width=3)
        for y_v in range(45, 185, 25):
            draw.line([(112, y_v), (160, y_v - 15)], fill=(70, 160, 70), width=2)
            draw.line([(112, y_v), (64, y_v - 15)], fill=(70, 160, 70), width=2)
            
        if pattern == "early_blight":
            for center in [(95, 75), (140, 130), (80, 145)]:
                cx, cy = center
                draw.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(185, 180, 45))
                draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(70, 45, 25))
                draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], outline=(100, 65, 35), width=2)
                draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(40, 25, 15))
        elif pattern == "late_blight":
            for box in [(115, 60, 175, 115), (60, 110, 115, 165), (120, 140, 165, 185)]:
                draw.ellipse(box, fill=(50, 40, 30))
                draw.ellipse((box[0]-4, box[1]-4, box[2]+4, box[3]+4), outline=(190, 205, 170), width=2)
        elif pattern == "rust":
            np.random.seed(42)
            for _ in range(35):
                rx = np.random.randint(65, 160)
                ry = np.random.randint(45, 180)
                draw.ellipse([rx, ry, rx + 6, ry + 4], fill=(185, 80, 25))
        elif pattern == "scab":
            for center in [(100, 90), (135, 105), (85, 130), (120, 160)]:
                cx, cy = center
                draw.ellipse([cx - 14, cy - 12, cx + 14, cy + 12], fill=(30, 45, 25))
        elif pattern == "bacterial_spot":
            for spot in [(85, 65), (130, 80), (100, 115), (145, 135), (75, 145), (115, 165)]:
                sx, sy = spot
                draw.ellipse([sx - 8, sy - 8, sx + 8, sy + 8], fill=(195, 195, 60))
                draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(45, 25, 15))
                
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
        img.save(img_path, format="JPEG", quality=92)
        print(f"Generated sample image: {img_path} ({class_name})")

def generate_dataset_metadata(output_csv_path: str):
    """
    Generate dataset metadata summary table for the 38 PlantVillage classes.
    """
    rows = []
    sample_counts = {
        "Tomato___Early_blight": 1000,
        "Tomato___Late_blight": 1909,
        "Tomato___healthy": 1591,
        "Tomato___Bacterial_spot": 2127,
        "Potato___Early_blight": 1000,
        "Potato___Late_blight": 1000,
        "Potato___healthy": 152,
        "Corn_(maize)___Common_rust_": 1192,
        "Corn_(maize)___healthy": 1162,
        "Apple___Apple_scab": 630,
        "Apple___Black_rot": 621,
        "Apple___Cedar_apple_rust": 275,
        "Apple___healthy": 1645,
        "Pepper,_bell___Bacterial_spot": 997,
        "Pepper,_bell___healthy": 1478
    }
    
    for cls in PLANTVILLAGE_CLASSES:
        info = get_disease_info(cls)
        count = sample_counts.get(cls, 1000)
        is_healthy = "healthy" in cls.lower()
        rows.append({
            "class_name": cls,
            "crop": info["crop"],
            "disease": info["disease"],
            "is_healthy": is_healthy,
            "pathogen": info["pathogen"],
            "severity_level": info["severity"],
            "sample_count": count
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Dataset metadata saved to: {output_csv_path}")
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_dir = os.path.join(base_dir, "data", "samples")
    meta_path = os.path.join(base_dir, "data", "dataset_metadata.csv")
    generate_benchmark_samples(samples_dir)
    generate_dataset_metadata(meta_path)
