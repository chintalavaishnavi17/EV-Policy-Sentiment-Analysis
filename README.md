# Public Sentiment Analysis on India's Electric Vehicle (EV) Policy

## Project Overview

This project analyzes public sentiment toward India's Electric Vehicle (EV) policy using YouTube comments.

The system combines:
- YouTube Data API for collecting public comments
- Text preprocessing and policy-relevance filtering
- Weakly supervised sentiment labeling
- TF-IDF + Linear SVM
- Fine-tuned DistilBERT
- Policy-theme analysis
- Flask web application
- Interactive dashboard

## Objectives

1. Collect public YouTube comments related to India's EV policy.
2. Clean and filter comments for India-specific EV-policy relevance.
3. Generate candidate sentiment labels using a pretrained sentiment model.
4. Compare TF-IDF + SVM with fine-tuned DistilBERT.
5. Analyze sentiment across important EV-policy themes.
6. Build a web application for real-time sentiment prediction.

## Dataset

The final modeling dataset contains **2,737 unique India EV-policy-related YouTube comments**.

The comments were collected from 207 unique videos and 138 unique YouTube channels.

### Candidate Label Distribution

| Sentiment | Comments | Percentage |
|---|---:|---:|
| Positive | 212 | 7.75% |
| Neutral | 1,834 | 67.01% |
| Negative | 691 | 25.25% |
| Total | 2,737 | 100% |

## Methodology

### Data Collection

Public YouTube comments were collected using the YouTube Data API v3.

Targeted searches covered general EV policy, PM E-DRIVE, FAME, subsidies, charging infrastructure, GST and taxation, EV manufacturing, battery technology, EV adoption, and policy implementation.

### Text Preprocessing

The comments were cleaned by removing URLs, user mentions, empty comments, duplicate comments, extremely short comments, and comments that were not relevant to India's EV policy.

### Candidate Sentiment Labeling

Candidate sentiment labels were generated using the pretrained model `cardiffnlp/twitter-roberta-base-sentiment-latest`.

These labels are treated as **weakly supervised / model-generated candidate labels**, rather than manually verified ground-truth labels.

## Machine Learning Models

### TF-IDF + Linear SVM

TF-IDF was used to convert text into numerical features, followed by a Linear Support Vector Machine classifier.

### DistilBERT

A pretrained `distilbert-base-uncased` transformer was fine-tuned for three-class sentiment classification.

The dataset was divided into 80% training, 10% validation, and 10% testing using stratified splitting.

Both models were evaluated on the same held-out test set.

## Results

### Model Comparison

| Metric | TF-IDF + SVM | DistilBERT |
|---|---:|---:|
| Accuracy | 75.91% | **86.86%** |
| Macro Precision | 68.15% | **84.64%** |
| Macro Recall | 65.83% | **81.62%** |
| Macro F1 | 66.88% | **83.05%** |

DistilBERT outperformed the TF-IDF + SVM baseline across all reported evaluation metrics on the same held-out test set.

The improvement in Macro F1 was approximately **16.17 percentage points**.

## Policy Theme Analysis

The project analyzes sentiment across the following themes:

1. EV Adoption
2. Policy Implementation
3. Battery & Technology
4. GST & Taxation
5. Charging Infrastructure
6. EV Manufacturing
7. FAME
8. Subsidies & Incentives
9. PM E-DRIVE

Neutral sentiment is dominant across most themes.

Comparatively higher negative sentiment was observed around EV adoption, policy implementation, battery and technology, GST and taxation, and charging infrastructure.

These findings describe the collected YouTube-comment sample and should not be interpreted as representative of the entire Indian population.

## Web Application

The Flask application contains three main sections:

### Sentiment Analyzer

Users can enter an EV-policy-related comment and receive a predicted sentiment and model confidence score.

### Dashboard

The dashboard presents overall sentiment distribution and policy-theme sentiment analysis.

### Model Comparison

The comparison page presents TF-IDF + SVM and DistilBERT performance using Accuracy, Macro Precision, Macro Recall, and Macro F1.

## Project Structure

```text
EV-Policy-Sentiment-Analysis/
|-- app.py
|-- requirements.txt
|-- README.md
|-- templates/
|   |-- index.html
|   |-- dashboard.html
|   |-- comparison.html
|-- static/
|   |-- style.css
|-- data/
|   |-- ev_policy_modeling_dataset.csv
|-- results/
|   |-- dashboard_data.json
|   |-- overall_sentiment_summary.csv
|   |-- policy_theme_sentiment_summary.csv
|-- model/
|   |-- ev_distilbert_final/
|-- notebook/
|   |-- EV_Policy_Sentiment_Analysis_Final.ipynb
```

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in a browser.

## Limitations

1. Sentiment labels are model-generated candidate labels and are not fully human-annotated ground truth.
2. YouTube comments may not represent the views of the entire Indian population.
3. Sarcasm, mixed sentiment, slang, and ambiguous comments can be difficult to classify.
4. Policy-theme assignment uses keyword/rule-based matching.
5. Model confidence scores should not be interpreted as calibrated probabilities of correctness.
6. Themes with fewer comments should be interpreted cautiously.

## Future Improvements

- Human-validated sentiment annotations
- Larger and more balanced datasets
- Multilingual and code-mixed Indian-language sentiment analysis
- Advanced sarcasm detection
- More robust topic modeling
- Time-based sentiment trend analysis
- Better calibrated confidence estimates
- Additional public data sources

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, PyTorch, Hugging Face Transformers, DistilBERT, Flask, YouTube Data API v3, HTML, CSS, JavaScript, and Chart.js.

## Research Statement

This project investigates how transformer-based language models can be used to analyze public online sentiment toward India's EV policy and compares their performance with a traditional TF-IDF + SVM text-classification approach.

The results indicate that the fine-tuned DistilBERT model achieved stronger performance than the traditional baseline on the held-out test set used in this study.