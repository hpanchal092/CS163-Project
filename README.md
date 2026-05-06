# 🚀 CS 163 Data Science Project  
## Early Prediction of E-commerce Product Success  

**Group ID:** 13  
**Members:** Vince Lai, Harsh Panchal  
**Course:** CS 163 – Data Science Senior Project  

🌐 **Live Website:**  
https://cs163-g13-vince.uw.r.appspot.com  

☁️ **Cloud Run API:**  
https://product-success-api-14643848899.us-west1.run.app  

---

## 📌 Overview

This project predicts whether an e-commerce product will become successful using **only the first 4 weeks of review data**.

👉 Success is defined as:
> Top 10% of products in the same category based on lifetime review count

🎯 Goal:
- Help sellers identify **high-potential products early**
- Avoid waiting months for full sales data

---

## 💡 Motivation

Online sellers often face uncertainty:
- Which product should I invest in?
- Which product will succeed?

Instead of waiting months, we use:
- Early review behavior
- Engagement signals

➡️ To predict future success quickly

---

## 📊 Dataset

We use the **Amazon Reviews Dataset (UCSD / McAuley Lab)**  

🔗 https://nijianmo.github.io/amazon/index.html  

### Categories Used:
- Electronics  
- Clothing, Shoes, Jewelry  
- Tools & Home Improvement  
- Toys & Games  
- Sports & Outdoors  

### Sampling:
- 10,000 products per category  
- 50,000 total products  
- Label: Top 10% = Successful  

📦 Stored in Google Cloud Storage:
gs://cs163-g13-product-data-vince/project_dataset_summary.csv


---

## ⚙️ Project Pipeline

Raw Data
↓
Sampling
↓
Feature Engineering (First 4 Weeks)
↓
EDA
↓
Model Training
↓
Evaluation
↓
Website + Cloud Deployment


---

## 🧠 Feature Engineering

We extract features from **first 4 weeks only**:

- Review count
- Reviews per day (velocity)
- Mean / median / min / max rating
- Rating standard deviation
- Helpful votes
- Votes per review
- Review text length
- Product category

---

## 🤖 Models

### 1. Logistic Regression
- Simple baseline
- Easy to interpret

### 2. XGBoost
- Captures nonlinear relationships
- Handles feature interactions better

---

## 📈 Evaluation Metrics

Because dataset is imbalanced (Top 10%):

- ROC-AUC  
- Average Precision  
- Precision / Recall / F1  
- ⭐ Precision@K (most important)

👉 We treat this as a **ranking problem**, not just classification.

---

## 📊 Results

| Model | ROC-AUC | Precision@K |
|------|--------|------------|
| Logistic Regression | ~0.638 | ~0.257 |
| XGBoost | ~0.679 | ~0.287 |

✅ XGBoost performs better across most metrics

---

## 🔍 Key Findings

### 1. Early data is useful
First 4 weeks already contain predictive signals

### 2. Engagement > Rating
Better predictors:
- Review count
- Votes
- Velocity

Ratings alone are weak (most products are already 4–5⭐)

### 3. Category matters
Performance varies by product category

---

## 🌐 Website Features

Built with **Dash + Plotly**

Pages:
- Home
- EDA
- Methods
- ML Models
- Findings
- System Design

Includes:
- 📊 Visualizations
- 📈 Interactive charts
- 🤖 Model results
- ☁️ Cloud integration

---

## ☁️ Cloud Deployment

### Google App Engine
Website hosting: https://cs163-g13-vince.uw.r.appspot.com


### Google Cloud Storage
Dataset storage: gs://cs163-g13-product-data-vince/project_dataset_summary.csv


### Google Cloud Run
Model API: https://product-success-api-14643848899.us-west1.run.app


---

## 🔌 API Example

### Endpoint:
POST /predict

### Input

```json
{
  "n_reviews_4w": 10,
  "mean_rating_4w": 4.5,
  "reviews_per_day_4w": 0.5,
  "sum_vote_4w": 25
}
```

### Output

```json
{
  "success_probability": 0.62,
  "prediction": "High potential"
}
```

---

## System Architecture

```text
User
↓
Dash Website (App Engine)
↓
Visualizations + Results
↓
Cloud Run API
↓
Cloud Storage
```

**Clean separation:**

- Frontend: Dash website
- Backend: Cloud Run API
- Data: Google Cloud Storage

---

## Project Structure

```text
CS163-Project/
├── app.py
├── app.yaml
├── requirements.txt
├── README.md
├── assets/
├── cloud_data/
├── inference_service/
├── eda.ipynb
└── prelim_results_models.ipynb
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/hpanchal092/CS163-Project.git
cd CS163-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

### 4. Open in browser

```text
http://127.0.0.1:8050
```

---

## Future Work

- Add NLP sentiment features
- Try different time windows, such as 2, 4, and 8 weeks
- Train category-specific models
- Improve the success label definition
- Connect the website UI directly to the Cloud Run API

---

## Conclusion

Early review data is useful but not perfect. Engagement signals outperform rating-only features, and XGBoost gives the best performance.

**Best use case:** ranking high-potential products early.