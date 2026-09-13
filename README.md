# Public Sentiment Analysis on India's Electric Vehicle (EV) Policy

## Live Demo

**Live Application:**  
https://ev-policy-sentiment-analysis.onrender.com

The deployed Flask application provides:

- Real-time EV-policy sentiment prediction
- Overall sentiment dashboard
- Policy-theme sentiment analysis
- TF-IDF + SVM vs DistilBERT model comparison

The application is deployed on Render using a CPU-optimized quantized ONNX version of the fine-tuned DistilBERT model.

---

## Project Overview

This project analyzes public sentiment toward India's Electric Vehicle (EV) policy using YouTube comments.

The system combines:

- YouTube Data API v3 for collecting public comments
- Text preprocessing and policy-relevance filtering
- Weakly supervised sentiment labeling
- TF-IDF + Linear SVM
- Fine-tuned DistilBERT
- Policy-theme analysis
- Flask web application
- Interactive dashboard

The primary objective is to compare a traditional machine-learning text classification approach with a transformer-based language model for EV-policy sentiment analysis.

---

## Objectives

1. Collect public YouTube comments related to India's EV policy.
2. Clean and filter comments for India-specific EV-policy relevance.
3. Generate candidate sentiment labels using a pretrained sentiment model.
4. Compare TF-IDF + SVM with fine-tuned DistilBERT.
5. Analyze sentiment across important EV-policy themes.
6. Build a web application for real-time sentiment prediction.

---

## Dataset

The final modeling dataset contains **2,737 unique India EV-policy-related YouTube comments**.

The comments were collected from:

- **207 unique videos**
- **138 unique YouTube channels**

### Candidate Label Distribution

| Sentiment | Comments | Percentage |
|-----------|----------|------------|
| Positive | 212 | 7.75% |
| Neutral | 1,834 | 67.01% |
| Negative | 691 | 25.25% |
| **Total** | **2,737** | **100%** |

### Important Note on Labels

The sentiment labels used for modeling are **weakly supervised / model-generated candidate labels**.

They were generated using the pretrained:

`cardiffnlp/twitter-roberta-base-sentiment-latest`

model.

Therefore, these labels should not be interpreted as independently human-verified ground-truth annotations.

---

## Methodology

### 1. Data Collection

Public YouTube comments were collected using the **YouTube Data API v3**.

Targeted searches covered major areas of India's EV policy, including:

- General EV policy
- PM E-DRIVE
- FAME / FAME 2
- EV subsidies and incentives
- Charging infrastructure
- GST and taxation
- EV manufacturing and PLI
- Battery technology and battery policy
- EV adoption
- Policy implementation
- Policy-related criticism and benefits

---

### 2. Text Preprocessing

The collected comments were processed to improve dataset quality.

The preprocessing pipeline included:

- HTML entity decoding
- URL removal
- User mention removal
- Whitespace normalization
- Empty-comment removal
- Duplicate removal
- Removal of extremely short comments
- Policy-relevance filtering
- India-specific EV-policy filtering

After preprocessing and filtering, **2,737 comments** were retained for the final modeling dataset.

---

### 3. Candidate Sentiment Labeling

Candidate sentiment labels were generated using:

`cardiffnlp/twitter-roberta-base-sentiment-latest`

The original model outputs were mapped into three sentiment classes:

- **Positive**
- **Neutral**
- **Negative**

These labels were treated as **silver / weakly supervised labels** rather than manually verified ground truth.

This approach allowed the project to construct a usable labeled dataset without requiring hundreds or thousands of manual annotations.

---

## Machine Learning Models

### TF-IDF + Linear SVM

The first approach uses a traditional text-classification pipeline.

1. TF-IDF converts text into numerical feature vectors.
2. A Linear Support Vector Machine (SVM) performs sentiment classification.

This model serves as the traditional machine-learning baseline.

---

### DistilBERT

The second approach uses:

`distilbert-base-uncased`

The pretrained DistilBERT model was fine-tuned for three-class sentiment classification:

- Negative
- Neutral
- Positive

The dataset was divided using a stratified split into:

- **80% Training**
- **10% Validation**
- **10% Testing**

Both models were evaluated on the **same held-out test set**.

Test set size:

**274 comments**

---

## Results

### Model Comparison

| Metric | TF-IDF + SVM | DistilBERT |
|--------|--------------|------------|
| Accuracy | 75.91% | **86.86%** |
| Macro Precision | 68.15% | **84.64%** |
| Macro Recall | 65.83% | **81.62%** |
| Macro F1 | 66.88% | **83.05%** |

DistilBERT outperformed the TF-IDF + SVM baseline across all reported evaluation metrics on the same held-out test set.

### Improvement

The Macro F1 score improved from:

**66.88% → 83.05%**

This represents an improvement of approximately:

**16.17 percentage points**

The improvement was also observed in accuracy, macro precision, and macro recall.

---

## Per-Class Performance

### TF-IDF + SVM

| Sentiment | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| Negative | 60.00% | 60.87% | 60.43% |
| Neutral | 83.33% | 84.24% | 83.78% |
| Positive | 61.11% | 52.38% | 56.41% |

### DistilBERT

| Sentiment | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| Negative | 80.30% | 76.81% | 78.52% |
| Neutral | 89.42% | 91.85% | 90.62% |
| Positive | 84.21% | 76.19% | 80.00% |

DistilBERT showed stronger performance across all three sentiment classes, with particularly noticeable improvements for Negative and Positive sentiment.

---

## Policy Theme Analysis

The project analyzes sentiment across the following EV-policy themes:

1. EV Adoption
2. Policy Implementation
3. Battery & Technology
4. GST & Taxation
5. Charging Infrastructure
6. EV Manufacturing
7. FAME
8. Subsidies & Incentives
9. PM E-DRIVE

### Key Observations

Neutral sentiment is dominant across most policy themes.

Comparatively higher negative sentiment was observed around:

- EV adoption
- Policy implementation
- Battery and technology
- GST and taxation
- Charging infrastructure

The project does **not** interpret these results as evidence that the overall Indian population is negative toward EV policy.

The findings describe sentiment within the collected YouTube-comment sample only.

Themes with relatively small numbers of comments, particularly PM E-DRIVE, should be interpreted cautiously.

---

## Web Application

The project includes a Flask-based web application with three main sections.

### 1. Sentiment Analyzer

Users can enter an EV-policy-related comment and receive:

- Predicted sentiment
- Model confidence score

Example:

> "The government should provide more support for electric vehicles."

The application processes the input using the deployed DistilBERT model.

---

### 2. Dashboard

The dashboard provides:

- Overall sentiment distribution
- Positive / Neutral / Negative comment counts
- Policy-theme sentiment analysis
- Theme-level sentiment percentages

---

### 3. Model Comparison

The comparison page presents the performance of:

- TF-IDF + Linear SVM
- Fine-tuned DistilBERT

Metrics include:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

---

## Deployment

The application is deployed using:

- **Render** for Flask web hosting
- **Hugging Face** for model hosting
- **ONNX Runtime** for CPU inference
- **Quantized ONNX model** for reduced model size

### Hugging Face Models

#### Original Fine-Tuned DistilBERT

https://huggingface.co/VaishnaviC17/ev-policy-distilbert

This repository contains the original fine-tuned DistilBERT model used for the research experiments.

#### Quantized ONNX Deployment Model

https://huggingface.co/VaishnaviC17/ev-policy-distilbert-quantized

This repository contains the CPU-optimized quantized ONNX model used by the deployed Flask application.

Dynamic quantization reduced the model size substantially, making CPU-based deployment more practical.

---

## Project Structure

```text
EV-Policy-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── ev_policy_modeling_dataset.csv
│
├── results/
│   ├── dashboard_data.json
│   ├── overall_sentiment_summary.csv
│   └── policy_theme_sentiment_summary.csv
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── comparison.html
│
├── static/
│   └── style.css
│
└── notebook/
    └── EV_Policy_Sentiment_Analysis_Final.ipynb
