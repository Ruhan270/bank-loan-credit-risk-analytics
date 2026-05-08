"""
data_generator.py
Generates a realistic synthetic bank loan dataset for analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_loan_dataset(n=5000):
    """Generate synthetic bank loan dataset mimicking real-world distributions."""

    # --- Demographics ---
    ages = np.random.normal(42, 12, n).clip(21, 75).astype(int)
    incomes = np.random.lognormal(mean=11.0, sigma=0.5, size=n).clip(20000, 500000).round(-2)
    employment_years = np.random.exponential(scale=7, size=n).clip(0, 40).round(1)

    employment_type = np.random.choice(
        ['Salaried', 'Self-Employed', 'Business Owner', 'Retired', 'Part-Time'],
        size=n, p=[0.55, 0.20, 0.12, 0.08, 0.05]
    )

    education = np.random.choice(
        ['High School', 'Bachelor\'s', 'Master\'s', 'PhD', 'Associate'],
        size=n, p=[0.22, 0.45, 0.22, 0.06, 0.05]
    )

    # --- Credit Profile ---
    credit_scores = np.random.normal(680, 80, n).clip(300, 850).astype(int)
    num_credit_lines = np.random.poisson(4, n).clip(0, 20)
    delinquencies = np.random.poisson(0.3, n).clip(0, 10)
    debt_to_income = np.random.beta(2, 5, n).round(3)  # 0 to 1
    existing_loans = np.random.randint(0, 5, n)

    # --- Loan Details ---
    loan_amounts = np.random.lognormal(mean=10.5, sigma=0.8, size=n).clip(5000, 500000).round(-3)
    loan_purposes = np.random.choice(
        ['Home Improvement', 'Debt Consolidation', 'Auto', 'Education',
         'Medical', 'Business', 'Personal', 'Vacation'],
        size=n, p=[0.20, 0.30, 0.15, 0.10, 0.07, 0.10, 0.05, 0.03]
    )
    loan_terms = np.random.choice([12, 24, 36, 48, 60, 84], size=n,
                                   p=[0.05, 0.10, 0.30, 0.20, 0.25, 0.10])

    # Interest rates influenced by credit score
    base_rate = 12 - (credit_scores - 300) / 100
    interest_rates = (base_rate + np.random.normal(0, 1.5, n)).clip(3.5, 28.0).round(2)

    # --- Application Dates ---
    start_date = datetime(2019, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365*5)) for _ in range(n)]
    application_dates = pd.to_datetime(dates)

    # --- Loan Status (target variable) ---
    # Probability of default influenced by credit score, DTI, delinquencies
    default_prob = (
        0.05
        + 0.25 * (1 - (credit_scores - 300) / 550)
        + 0.20 * debt_to_income
        + 0.10 * (delinquencies / 10)
        + 0.05 * (existing_loans / 5)
    ).clip(0.02, 0.75)

    statuses = []
    for prob in default_prob:
        r = random.random()
        if r < prob * 0.4:
            statuses.append('Charged Off')
        elif r < prob:
            statuses.append('Late (31-120 days)')
        else:
            statuses.append('Fully Paid')

    df = pd.DataFrame({
        'loan_id': [f'LN{str(i).zfill(6)}' for i in range(1, n+1)],
        'application_date': application_dates,
        'age': ages,
        'annual_income': incomes,
        'employment_type': employment_type,
        'employment_years': employment_years,
        'education': education,
        'credit_score': credit_scores,
        'num_credit_lines': num_credit_lines,
        'delinquencies_2yr': delinquencies,
        'debt_to_income_ratio': debt_to_income,
        'existing_loans': existing_loans,
        'loan_amount': loan_amounts,
        'loan_purpose': loan_purposes,
        'loan_term_months': loan_terms,
        'interest_rate': interest_rates,
        'loan_status': statuses,
    })

    # Add derived features
    df['monthly_payment'] = (
        df['loan_amount'] * (df['interest_rate'] / 100 / 12) /
        (1 - (1 + df['interest_rate'] / 100 / 12) ** (-df['loan_term_months']))
    ).round(2)

    df['payment_to_income_ratio'] = (
        (df['monthly_payment'] * 12) / df['annual_income']
    ).round(4)

    df['is_default'] = df['loan_status'].isin(['Charged Off', 'Late (31-120 days)']).astype(int)
    df['year'] = df['application_date'].dt.year
    df['quarter'] = df['application_date'].dt.to_period('Q').astype(str)

    return df


if __name__ == "__main__":
    df = generate_loan_dataset(5000)
    df.to_csv('../data/loan_data.csv', index=False)
    print(f"Dataset generated: {df.shape}")
    print(df['loan_status'].value_counts())
