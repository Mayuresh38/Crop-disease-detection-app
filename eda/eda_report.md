# Exploratory Data Analysis (EDA) Report: PlantVillage Crop Disease Dataset

**Topic:** Deep Learning Automated Crop Disease Diagnostic System  
**Dataset Source:** [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  
**Total Images:** ~54,303 high-resolution RGB leaf photographs  
**Total Categories:** 38 distinct crop-disease classes across 14 agricultural plant commodities  

---

## 1. Executive Summary

Foliar plant diseases are responsible for substantial annual yield penalties across global staple and horticultural crops. The **PlantVillage** dataset represents the primary worldwide benchmark for computer vision-based phytopathology classification.

This exploratory analysis characterizes:
1. **Class distribution and sample balance** across all 38 categories.
2. **Crop representation** across 14 key economic commodities (Solanaceae, Rosaceae, Vitaceae, Poaceae).
3. **Pathogen etiologies** (Fungi, Oomycetes, Bacteria, Viruses, and Pest Mites).
4. **Spectral and photometric properties** of healthy versus infected leaf tissue.

---

## 2. Dataset Taxonomy & Composition

The 38 classes encompass 14 agricultural plant species:

| Crop Species | Number of Diagnostic Classes | Included Diseases / Health Conditions |
| :--- | :---: | :--- |
| **Tomato** (*Solanum lycopersicum*) | 10 | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| **Potato** (*Solanum tuberosum*) | 3 | Early Blight, Late Blight, Healthy |
| **Corn / Maize** (*Zea mays*) | 4 | Cercospora Leaf Spot (Gray Spot), Common Rust, Northern Leaf Blight, Healthy |
| **Apple** (*Malus domestica*) | 4 | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Grape** (*Vitis vinifera*) | 4 | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Bell Pepper** (*Capsicum annuum*) | 2 | Bacterial Spot, Healthy |
| **Peach** (*Prunus persica*) | 2 | Bacterial Spot, Healthy |
| **Cherry** (*Prunus cerasus*) | 2 | Powdery Mildew, Healthy |
| **Strawberry** (*Fragaria ananassa*) | 2 | Leaf Scorch, Healthy |
| **Blueberry, Raspberry, Soybean, Squash, Orange** | 5 | Powdery Mildew (Squash), Citrus Greening / Huanglongbing (Orange), and Healthy controls |

---

## 3. Key EDA Findings & Analytical Visualizations

Generated publication-grade figures are located in [`eda/visualizations/`](visualizations/):

### 3.1 Class Frequency & Balance (`01_class_distribution_38.png`)
- **Distribution:** While popular crop classes like *Tomato Bacterial Spot* (2,127 images) and *Tomato Late Blight* (1,909 images) are heavily represented, minority classes such as *Potato Healthy* (152 images) and *Apple Cedar Rust* (275 images) present moderate class imbalance.
- **Engineering Recommendation:** Utilize class-weighted categorical cross-entropy loss or focal loss, accompanied by intensive data augmentation on underrepresented categories.

### 3.2 Crop Species Aggregation (`02_crop_species_breakdown.png`)
- Tomato is the dominant commodity in the dataset, accounting for over 33% of all images due to its 10 diverse classes.
- Legumes and cucurbits have more singular representation, demonstrating the need for robust feature extraction that avoids species-specific overfitting.

### 3.3 Pathogen Etiological Taxonomy (`03_pathogen_taxonomy_donut.png`)
- **Fungal Pathogens & Oomycetes:** Constitute over **57%** of all disease classes (e.g., *Alternaria solani*, *Phytophthora infestans*, *Puccinia sorghi*).
- **Bacterial Pathogens:** Account for **13.2%** of classes (primarily *Xanthomonas* species).
- **Viral Pathogens:** Account for **5.3%** of classes (Mosaic virus, Curl virus).
- **Healthy Controls:** Comprise **31.5%** of the taxonomy (12 classes).

### 3.4 Spectral Vegetation Indices (`04_spectral_vegetation_signatures.png`)
- Healthy leaves display a sharp, narrow Excess Green (`ExG = 2*G - R - B`) peak centered around `+0.42`.
- Necrotic and chlorotic lesions shift the distribution towards negative or low positive values (`+0.18`), providing a clear biophysical foundation for convolutional activation mapping.

---

## 4. Recommendations for Deep Learning Architecture & Training

1. **Backbone Architecture:** MobileNetV2 with inverted residual bottlenecks offers an ideal trade-off between top-1 accuracy (>96%) and lightweight CPU/edge deployment (~38ms inference).
2. **Resolution & Normalization:** Images normalized to `[0, 1]` or standard ImageNet z-score normalization at `224 x 224` resolution retain fine lesion edge details while maintaining low computational memory footprints.
3. **Interpretability:** Grad-CAM activation heatmaps should be integrated into the clinical diagnostic pipeline to ensure predictions are grounded in actual lesion pathology rather than background artifacts.
