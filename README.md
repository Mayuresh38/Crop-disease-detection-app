# AgriShield AI: Crop Disease Detection & Agronomic Decision Support System

An end-to-end Computer Vision & Phytopathology AI application that identifies 38 distinct crop diseases across 14 agricultural plant commodities from leaf photography. Features Grad-CAM explainability, biophysical vegetation index extraction, micro-climate disease spread forecasting, and targeted organic/chemical treatment regimens.

---

## 🌟 Key Features

1. **38-Class Deep Learning Diagnostic Engine:** Trained on the benchmark [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) utilizing MobileNetV2 Transfer Learning.
2. **Grad-CAM Visual Heatmaps:** Highlights the exact infected foliar lesion regions guiding the convolutional neural network's diagnosis.
3. **Agronomic Treatment Protocols:** Provides actionable recommendations:
   - Bio-fungicides & biological agents (e.g., *Bacillus subtilis*, neem extract)
   - Labeled chemical actives (e.g., Chlorothalonil, Mancozeb, Copper Hydroxide)
   - Cultural practices (drip irrigation, canopy aeration, 3-year crop rotation)
4. **Biophysical Spectral Indices:** Computes Excess Green (ExG), Green Leaf Index (GLI), and necrosis surface area percentage.
5. **Micro-Climate Spread Forecaster:** Simulates ambient temperature, relative humidity, and rainfall to calculate localized epidemiological infection risk.
6. **Dedicated EDA Module (`eda/`):** Automated quantitative analysis with publication-grade charts and summary markdown report.

---

## 📂 Project Architecture

```text
crop_disease_detector/
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Python dependency specifications
├── README.md                  # Project overview & documentation
├── DEPLOYMENT.md              # Cloud & Containerized deployment guide
├── project_explanation.md     # Architectural & algorithmic deep dive
├── eda/                       # Exploratory Data Analysis Module
│   ├── eda_analysis.py        # Automated EDA generator script
│   ├── eda_report.md          # Comprehensive analytical report
│   └── visualizations/        # High-resolution PNG figures (01-04)
├── src/                       # Modular source code
│   ├── data_loader.py         # Data ingestion & 38-class knowledge base
│   ├── preprocessing.py       # Tensor normalization & augmentation
│   ├── features.py            # Spectral indices & epidemiological risk
│   ├── models.py              # MobileNetV2 & Compact CNN architectures
│   ├── train.py               # Model training & artifact serialization
│   └── explainability.py      # Grad-CAM heatmap implementation
├── models/                    # Serialized models & metadata JSONs
│   ├── crop_disease_model.keras
│   ├── class_indices.json
│   └── disease_info.json
├── data/                      # Dataset metadata & benchmark samples
│   ├── dataset_metadata.csv
│   └── samples/               # Preset test leaf images (.jpg)
└── tests/                     # Automated unit & integration tests
    └── test_pipeline.py
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate EDA Visualizations & Baseline Artifacts
```bash
python eda/eda_analysis.py
python src/data_loader.py
```

### 3. Run Unit Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
