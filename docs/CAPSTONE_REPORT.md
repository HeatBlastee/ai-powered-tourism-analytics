# AIML Capstone Project Report
**Project Title:** Preserving Heritage: Enhancing Tourism with AI
**Date:** January 25, 2026

---

## 1. Executive Summary
This project addresses two critical challenges in the tourism sector:
1.  **Automated Heritage Preservation**: Using Deep Learning to classify historical structures for maintenance monitoring.
2.  **Personalized Tourism Marketing**: Understanding tourist behavior and recommending personalized destinations using Data Science.

I built the solution using **TensorFlow** (EfficientNetB0) for vision tasks and **Collaborative Filtering** for recommendations, deployed as a scalable Enterprise API.

---

## Part 1: Deep Learning (Computer Vision)
**Objective**: Build an automated AI model to predict the category of a structure in an image.

### 1.1 Data Visualization
I analyzed the `Structures_dataset` containing images of historical monuments. I plotted sample images using OpenCV to verify class diversity.
*(Note: See `notebooks/EDA_Vision.ipynb` for visual plots).*

### 1.2 Model Architecture Selection
I selected **EfficientNetB0** over VGG16/ResNet50 because:
*   **Accuracy vs. Efficiency**: EfficientNet achieves higher accuracy with significantly fewer parameters (5.3M vs 138M for VGG16), making it ideal for deployment.
*   **Transfer Learning**: I loaded pre-trained `ImageNet` weights to leverage feature extraction capabilities.

### 1.3 Architecture Configuration
I customized the top layers for the specific classification task:
*   **Global Average Pooling**: Reduced spatial dimensions.
*   **Dense Layer**: 1024 units with `ReLU` activation.
*   **Dropout**: I added `0.5` dropout rate to prevent overfitting.
*   **Output Layer**: `Softmax` activation for multi-class classification.

### 1.4 Training Strategy
*   **Optimizer**: Adam with learning rate `0.001`.
*   **Loss Function**: `Categorical Crossentropy`.
*   **Callbacks**: `EarlyStopping` (patience=3) to stop training when validation accuracy plateaued.
*   **Augmentation**: I applied Random Flip, Rotation, and Zoom to increase model robustness.

**Results:**
*   **Accuracy**: I achieved >85% validation accuracy.
*   **Observations**: The model initially overfitted, but Data Augmentation and Dropout stabilized the learning curve.

---

## Part 2: Data Science (Recommendation Engine)
**Objective**: Perform EDA and develop a recommendation engine.

### 2.1 Exploratory Data Analysis (EDA)
I analyzed `user.csv`, `tourism_rating.csv`, and `tourism_with_id.xlsx`.

#### I. Data Inspection
*   **Missing Values**: I handled these using a `dropna()` pipeline.
*   **Anomalies**: I filtered out ratings outside the 1-5 range.

#### II. User Analysis
*   **Age Distribution**: I found the average visitor age is **28.7 years** (Range: 18-40), indicating a young, tech-savvy demographic.
*   **Origin**: I discovered the majority of tourists come from **Bekasi (Jawa Barat)** and **Semarang**.

#### III. Place Category Analysis
*   **Categories Identified**: `Budaya` (Culture), `Taman Hiburan` (Amusement Parks), `Cagar Alam` (Nature Reserves), `Bahari` (Marine/Beach), `Pusat Perbelanjaan` (Shopping), `Tempat Ibadah` (Worship).
*   **Nature Enthusiasts**: I identified **Bandung** and **Yogyakarta** as the top cities for Nature enthusiasts.
*   **Most Popular Category**: **Taman Hiburan** (3053 ratings), followed by **Budaya** (2683 ratings).

#### IV. Top Rated Spots
My analysis shows the most loved tourist destinations are:
1.  **Keraton Surabaya** (~3.97)
2.  **Puncak Gunung Api Purba** (~3.88)
3.  **Kampung Cina** (~3.84)

### 2.2 Recommendation Model
I built a **Collaborative Filtering** system.
*   **Method**: Item-Based Filtering using user vectors.
*   **Algorithm**: **FAISS** (Facebook AI Similarity Search) for scalable similarity matching.
*   **Logic**: If a user likes "Borobudur", my system finds other items with similar user interaction vectors.

---

## 3. Improvements Summary
To meet the requirement of "XYZ Pvt. Ltd.", I went beyond a notebook and delivered a production-grade system:
*   **API**: I built a FastAPI service for real-time inference.
*   **UI**: I created a modern **Next.js** Dashboard for easy stakeholder demonstration.
*   **MLOps**: I integrated MLflow for experiment tracking and model versioning.

