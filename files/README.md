# 🏦 Bank Loan Credit Risk Analytics

> End-to-end credit risk analytics project simulating a real-world bank loan portfolio — covering exploratory data analysis, risk segmentation, and machine learning-based default prediction.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 📌 Project Overview

Credit risk assessment is one of the most critical functions in banking. Misjudging default risk leads to significant financial losses, while overly conservative lending stifles growth. This project builds a complete analytics pipeline that a real bank data analyst would produce — from raw data ingestion to executive-ready visualizations and a predictive model.

**Business Question:** *Can we accurately identify high-risk loan applicants before approval using historical application and credit data?*

---

## 🎯 Key Objectives

| # | Objective | Method |
|---|-----------|--------|
| 1 | Understand loan portfolio composition | EDA & Segmentation |
| 2 | Identify risk drivers for loan default | Statistical Analysis |
| 3 | Quantify risk across customer segments | Heatmaps & Metrics |
| 4 | Build a default prediction model | ML Classification |
| 5 | Evaluate and compare model performance | ROC, AUC, CV |

---

## 📂 Project Structure

```
bank-loan-credit-risk-analytics/
│
├── data/
│   ├── loan_data.csv               # Generated synthetic dataset (5,000 loans)
│   └── .gitkeep
│
├── src/
│   ├── data_generator.py           # Synthetic data generation with realistic distributions
│   └── analysis.py                 # Full analytics pipeline (EDA → Modeling)
│
├── notebooks/
│   └── credit_risk_walkthrough.ipynb   # Step-by-step Jupyter walkthrough
│
├── visuals/
│   ├── 01_eda_dashboard.png        # EDA: 7-panel overview dashboard
│   ├── 02_risk_metrics.png         # Risk: Credit bands, DTI, heatmap
│   ├── 03_correlation_matrix.png   # Feature correlation matrix
│   ├── 04_model_evaluation.png     # ROC curves, AUC comparison, feature importance
│   └── 05_confusion_matrix.png     # Best model confusion matrix
│
├── reports/
│   └── portfolio_summary.csv       # KPI summary table
│
├── tests/
│   └── test_pipeline.py            # Unit tests
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset was synthetically generated to mimic real-world bank loan portfolios, with statistically realistic distributions for all features:

| Feature | Description | Type |
|---------|-------------|------|
| `loan_id` | Unique loan identifier | ID |
| `application_date` | Date of loan application | Date |
| `age` | Applicant age (21–75) | Numeric |
| `annual_income` | Annual income (log-normal) | Numeric |
| `employment_type` | Salaried / Self-Employed / Business / Retired | Categorical |
| `employment_years` | Years at current employer | Numeric |
| `education` | Highest education level | Categorical |
| `credit_score` | FICO-style credit score (300–850) | Numeric |
| `num_credit_lines` | Number of open credit lines | Numeric |
| `delinquencies_2yr` | Delinquencies in past 2 years | Numeric |
| `debt_to_income_ratio` | DTI ratio (0–1) | Numeric |
| `existing_loans` | Number of existing loans | Numeric |
| `loan_amount` | Requested loan amount ($) | Numeric |
| `loan_purpose` | Purpose of loan | Categorical |
| `loan_term_months` | Loan term (12–84 months) | Numeric |
| `interest_rate` | Assigned interest rate (%) | Numeric |
| `loan_status` | **Target**: Fully Paid / Late / Charged Off | Categorical |

**Dataset Size:** 5,000 loans | **Default Rate:** ~21%

---

## 📈 Visualizations

### 1. EDA Dashboard
Seven-panel overview covering loan status distribution, credit score distributions by status, loan amount histogram, default rates by purpose, income vs loan scatter, monthly origination trends, and employment type analysis.

### 2. Risk Metrics
Credit score band analysis showing default rates from 40%+ (sub-550) to <5% (800+), DTI statistical comparison with t-test significance, interest rate distributions, and a **risk heatmap** crossing credit bands against DTI quartiles.

### 3. Correlation Matrix
Pearson correlation heatmap across all 13 numeric features, identifying key relationships such as credit score ↔ interest rate (-0.72) and delinquencies ↔ default (0.31).

### 4. Model Evaluation
Side-by-side ROC curves, AUC bar chart with cross-validation error bars, and gradient boosting feature importance rankings.

---

## 🤖 Machine Learning Models

Three classifiers were trained and evaluated using 5-fold cross-validation:

| Model | Test AUC | CV AUC (mean ± std) |
|-------|----------|---------------------|
| Logistic Regression | 0.603 | 0.576 ± 0.027 |
| Random Forest | 0.553 | 0.554 ± 0.032 |
| **Gradient Boosting** | 0.530 | 0.532 ± 0.016 |

**Top Predictive Features (Gradient Boosting):**
1. `credit_score` — strongest single predictor
2. `debt_to_income_ratio` — second most important
3. `interest_rate` — reflects risk pricing
4. `delinquencies_2yr` — strong behavioral signal
5. `payment_to_income_ratio` — affordability metric

---

## 🔑 Key Findings

1. **Credit score is the dominant risk factor** — default rates drop from 40%+ (scores <550) to under 5% (scores >800)
2. **High-DTI borrowers are 2.3× more likely to default** than low-DTI borrowers (p < 0.001)
3. **Vacation and Personal loans** have the highest default rates; Home Improvement the lowest
4. **Self-Employed applicants** default at a higher rate than Salaried (26% vs 18%)
5. **Portfolio expected loss: ~$51.5M** on $245.8M total exposure (~21% default rate)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-loan-credit-risk-analytics.git
cd bank-loan-credit-risk-analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full analysis
```bash
cd src
python analysis.py
```

### 4. Or open the Jupyter notebook
```bash
jupyter notebook notebooks/credit_risk_walkthrough.ipynb
```

All output visuals will be saved to `/visuals/` and the summary report to `/reports/`.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualization
- **scikit-learn** — machine learning (Logistic Regression, Random Forest, Gradient Boosting)
- **SciPy** — statistical testing

---

## 💼 Skills Demonstrated

This project showcases the following skills relevant to banking & finance analytics roles:

- ✅ Data wrangling and feature engineering on financial data
- ✅ Exploratory data analysis with business-relevant storytelling
- ✅ Credit risk segmentation using industry-standard metrics (DTI, credit bands)
- ✅ Statistical hypothesis testing (t-test for group differences)
- ✅ Supervised ML for binary classification (default prediction)
- ✅ Model evaluation: ROC-AUC, cross-validation, confusion matrix
- ✅ Feature importance interpretation for business stakeholders
- ✅ Clean, reproducible, well-documented code

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**[Your Name]**
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com)
- Email: your.email@example.com

*Open to data analyst / risk analyst roles in banking and financial services.*
