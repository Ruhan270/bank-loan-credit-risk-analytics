"""
analysis.py
Full end-to-end bank loan risk analytics pipeline.
Covers EDA, risk metrics, segmentation, and predictive modeling.
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_loan_dataset

# ── Color Palette ────────────────────────────────────────────────────────────
NAVY    = '#0D1B2A'
GOLD    = '#C9A84C'
BLUE    = '#1A6B8A'
TEAL    = '#2D9B8A'
RED     = '#C0392B'
GREY    = '#BDC3C7'
WHITE   = '#FAFAFA'
ACCENT  = '#E67E22'

plt.rcParams.update({
    'figure.facecolor': WHITE,
    'axes.facecolor':   '#F5F6FA',
    'axes.edgecolor':   '#CBD0D8',
    'axes.labelcolor':  NAVY,
    'axes.titlecolor':  NAVY,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.color':      NAVY,
    'ytick.color':      NAVY,
    'grid.color':       '#E2E5EB',
    'grid.linestyle':   '--',
    'grid.alpha':       0.7,
    'font.family':      'DejaVu Sans',
    'text.color':       NAVY,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'../visuals/{name}.png', dpi=150, bbox_inches='tight',
                facecolor=WHITE)
    plt.close(fig)
    print(f'  ✔ Saved {name}.png')


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n══ Loading data ══")
df = generate_loan_dataset(5000)
df.to_csv('../data/loan_data.csv', index=False)
print(f"Shape: {df.shape}  |  Columns: {list(df.columns)}")
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass balance:\n", df['loan_status'].value_counts())


# ════════════════════════════════════════════════════════════════════════════
# 2. EDA — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
print("\n══ EDA Dashboard ══")

fig = plt.figure(figsize=(20, 14))
fig.suptitle('Bank Loan Portfolio — Exploratory Data Analysis', fontsize=18,
             fontweight='bold', color=NAVY, y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# (A) Loan Status Distribution
ax = fig.add_subplot(gs[0, 0])
vc = df['loan_status'].value_counts()
colors_pie = [TEAL, RED, GOLD]
wedges, texts, autotexts = ax.pie(
    vc, labels=vc.index, autopct='%1.1f%%', startangle=140,
    colors=colors_pie, pctdistance=0.78,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for at in autotexts:
    at.set_fontsize(10); at.set_color('white'); at.set_fontweight('bold')
ax.set_title('Loan Status Distribution', fontweight='bold')

# (B) Credit Score Distribution
ax = fig.add_subplot(gs[0, 1])
for status, color in zip(['Fully Paid', 'Late (31-120 days)', 'Charged Off'],
                          [TEAL, GOLD, RED]):
    subset = df[df['loan_status'] == status]['credit_score']
    ax.hist(subset, bins=30, alpha=0.6, label=status, color=color, edgecolor='white')
ax.set_title('Credit Score by Loan Status', fontweight='bold')
ax.set_xlabel('Credit Score')
ax.set_ylabel('Count')
ax.legend(fontsize=8)
ax.axvline(df['credit_score'].mean(), color=NAVY, linestyle='--',
           linewidth=1.5, label='Mean')

# (C) Loan Amount Distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(df['loan_amount'] / 1000, bins=40, color=BLUE, edgecolor='white', alpha=0.85)
ax.set_title('Loan Amount Distribution', fontweight='bold')
ax.set_xlabel('Loan Amount ($K)')
ax.set_ylabel('Count')

# (D) Default Rate by Purpose
ax = fig.add_subplot(gs[1, :2])
purpose_default = (df.groupby('loan_purpose')['is_default']
                     .mean().sort_values(ascending=True) * 100)
bars = ax.barh(purpose_default.index, purpose_default.values,
               color=[RED if v > 20 else BLUE for v in purpose_default.values],
               edgecolor='white', height=0.65)
for bar, val in zip(bars, purpose_default.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, color=NAVY, fontweight='bold')
ax.set_title('Default Rate by Loan Purpose', fontweight='bold')
ax.set_xlabel('Default Rate (%)')
ax.axvline(df['is_default'].mean() * 100, color=GOLD, linestyle='--',
           linewidth=1.5, label=f"Portfolio Avg: {df['is_default'].mean()*100:.1f}%")
ax.legend(fontsize=9)

# (E) Income vs Loan Amount (scatter)
ax = fig.add_subplot(gs[1, 2])
sample = df.sample(600, random_state=1)
colors_map = {'Fully Paid': TEAL, 'Late (31-120 days)': GOLD, 'Charged Off': RED}
for status, grp in sample.groupby('loan_status'):
    ax.scatter(grp['annual_income']/1000, grp['loan_amount']/1000,
               c=colors_map[status], alpha=0.45, s=15, label=status)
ax.set_title('Income vs Loan Amount', fontweight='bold')
ax.set_xlabel('Annual Income ($K)')
ax.set_ylabel('Loan Amount ($K)')
ax.legend(fontsize=7, markerscale=1.5)

# (F) Loans Originated Over Time
ax = fig.add_subplot(gs[2, :2])
monthly = df.set_index('application_date').resample('ME')['loan_id'].count()
ax.fill_between(monthly.index, monthly.values, alpha=0.25, color=BLUE)
ax.plot(monthly.index, monthly.values, color=BLUE, linewidth=2)
ax.set_title('Monthly Loan Originations (2019–2024)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Loans')

# (G) Employment Type vs Default Rate
ax = fig.add_subplot(gs[2, 2])
emp_def = (df.groupby('employment_type')['is_default']
             .mean().sort_values(ascending=False) * 100)
ax.bar(emp_def.index, emp_def.values,
       color=[RED if v > 25 else BLUE for v in emp_def.values],
       edgecolor='white', width=0.6)
ax.set_title('Default Rate by Employment', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.set_xticklabels(emp_def.index, rotation=25, ha='right', fontsize=9)

save(fig, '01_eda_dashboard')


# ════════════════════════════════════════════════════════════════════════════
# 3. RISK METRICS
# ════════════════════════════════════════════════════════════════════════════
print("\n══ Risk Metrics ══")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Credit Risk Metrics & Segmentation', fontsize=17,
             fontweight='bold', color=NAVY)

# (A) Credit Score Bucket — Default Rates
ax = axes[0, 0]
bins = [300, 550, 620, 680, 740, 800, 850]
labels = ['<550\n(Very Poor)', '550–620\n(Poor)', '620–680\n(Fair)',
          '680–740\n(Good)', '740–800\n(Very Good)', '800+\n(Exceptional)']
df['credit_band'] = pd.cut(df['credit_score'], bins=bins, labels=labels)
band_stats = df.groupby('credit_band', observed=True)['is_default'].agg(['mean', 'count'])
band_stats['mean'] *= 100
colors_band = [RED if v > 30 else GOLD if v > 15 else TEAL for v in band_stats['mean']]
bars = ax.bar(band_stats.index, band_stats['mean'], color=colors_band,
              edgecolor='white', width=0.65)
for bar, (_, row) in zip(bars, band_stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{row['mean']:.1f}%\n(n={row['count']})",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=NAVY)
ax.set_title('Default Rate by Credit Score Band', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.set_xticklabels(band_stats.index, fontsize=8)

# (B) Debt-to-Income vs Default (box)
ax = axes[0, 1]
default_grp = df.groupby('is_default')['debt_to_income_ratio']
bp = ax.boxplot([default_grp.get_group(0), default_grp.get_group(1)],
                patch_artist=True, notch=True,
                boxprops=dict(facecolor=TEAL, alpha=0.7),
                medianprops=dict(color=GOLD, linewidth=2.5))
bp['boxes'][1].set_facecolor(RED)
ax.set_xticklabels(['Non-Default', 'Default'])
ax.set_title('Debt-to-Income Ratio: Default vs Non-Default', fontweight='bold')
ax.set_ylabel('Debt-to-Income Ratio')
t_stat, p_val = stats.ttest_ind(default_grp.get_group(0), default_grp.get_group(1))
ax.text(0.97, 0.97, f'p-value: {p_val:.2e}', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color=NAVY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=WHITE, edgecolor=GREY))

# (C) Interest Rate Distribution by Status
ax = axes[1, 0]
for status, color in zip(['Fully Paid', 'Late (31-120 days)', 'Charged Off'],
                          [TEAL, GOLD, RED]):
    data = df[df['loan_status'] == status]['interest_rate']
    ax.hist(data, bins=25, alpha=0.6, color=color, label=status, edgecolor='white')
ax.set_title('Interest Rate Distribution by Loan Status', fontweight='bold')
ax.set_xlabel('Interest Rate (%)')
ax.set_ylabel('Count')
ax.legend(fontsize=8)

# (D) Risk Score Heatmap — Credit Band × DTI Quartile
ax = axes[1, 1]
df['dti_quartile'] = pd.qcut(df['debt_to_income_ratio'], 4,
                              labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
heatmap_data = df.pivot_table(values='is_default', index='credit_band',
                               columns='dti_quartile', aggfunc='mean',
                               observed=True) * 100
sns.heatmap(heatmap_data, ax=ax, cmap='RdYlGn_r', annot=True, fmt='.1f',
            linewidths=0.5, linecolor=WHITE, cbar_kws={'label': 'Default Rate (%)'})
ax.set_title('Default Rate Heatmap\n(Credit Band × DTI Quartile)', fontweight='bold')
ax.set_xlabel('DTI Quartile')
ax.set_ylabel('Credit Score Band')

save(fig, '02_risk_metrics')


# ════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
print("\n══ Correlation Analysis ══")

num_cols = ['age', 'annual_income', 'employment_years', 'credit_score',
            'num_credit_lines', 'delinquencies_2yr', 'debt_to_income_ratio',
            'existing_loans', 'loan_amount', 'interest_rate',
            'monthly_payment', 'payment_to_income_ratio', 'is_default']

corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            square=True, linewidths=0.5, linecolor=WHITE,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'})
ax.set_title('Feature Correlation Matrix', fontsize=15, fontweight='bold', pad=15)
save(fig, '03_correlation_matrix')


# ════════════════════════════════════════════════════════════════════════════
# 5. PREDICTIVE MODELING — DEFAULT PREDICTION
# ════════════════════════════════════════════════════════════════════════════
print("\n══ Predictive Modeling ══")

# Feature Engineering
model_df = df.copy()
cat_cols = ['employment_type', 'education', 'loan_purpose']
le = LabelEncoder()
for col in cat_cols:
    model_df[col] = le.fit_transform(model_df[col])

features = ['age', 'annual_income', 'employment_years', 'employment_type',
            'education', 'credit_score', 'num_credit_lines',
            'delinquencies_2yr', 'debt_to_income_ratio', 'existing_loans',
            'loan_amount', 'loan_term_months', 'interest_rate',
            'payment_to_income_ratio', 'loan_purpose']

X = model_df[features]
y = model_df['is_default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                   random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                       learning_rate=0.05,
                                                       random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_s if name == 'Logistic Regression' else X_train
    X_te = X_test_s  if name == 'Logistic Regression' else X_test
    model.fit(X_tr, y_train)
    proba = model.predict_proba(X_te)[:, 1]
    pred  = model.predict(X_te)
    auc   = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    cv = cross_val_score(model, X_tr if name == 'Logistic Regression' else X_train,
                         y_train, cv=5, scoring='roc_auc')
    results[name] = {'model': model, 'proba': proba, 'pred': pred,
                     'auc': auc, 'fpr': fpr, 'tpr': tpr, 'cv': cv}
    print(f"  {name}: AUC={auc:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}")
    print(classification_report(y_test, pred, target_names=['Non-Default', 'Default']))

# ── Model Comparison Plot ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Credit Default Prediction — Model Evaluation', fontsize=16,
             fontweight='bold', color=NAVY)

# ROC Curves
ax = axes[0]
colors_model = [BLUE, TEAL, RED]
for (name, res), color in zip(results.items(), colors_model):
    ax.plot(res['fpr'], res['tpr'], color=color, linewidth=2.5,
            label=f"{name} (AUC = {res['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax.fill_between(results['Gradient Boosting']['fpr'],
                results['Gradient Boosting']['tpr'], alpha=0.08, color=RED)
ax.set_title('ROC Curves', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9)

# AUC Comparison Bar
ax = axes[1]
names  = list(results.keys())
aucs   = [results[n]['auc'] for n in names]
cv_std = [results[n]['cv'].std() for n in names]
bars = ax.bar(names, aucs, color=colors_model, edgecolor='white',
              width=0.5, yerr=cv_std, capsize=5,
              error_kw={'ecolor': NAVY, 'linewidth': 2})
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_title('AUC Score Comparison\n(with 5-fold CV std)', fontweight='bold')
ax.set_ylabel('AUC Score')
ax.set_xticklabels(names, rotation=15, ha='right')

# Feature Importance (Gradient Boosting)
ax = axes[2]
gb_model = results['Gradient Boosting']['model']
imp = pd.Series(gb_model.feature_importances_, index=features).sort_values()
top_imp = imp.tail(12)
bars = ax.barh(top_imp.index, top_imp.values,
               color=[RED if v > 0.08 else BLUE for v in top_imp.values],
               edgecolor='white', height=0.7)
ax.set_title('Top Feature Importances\n(Gradient Boosting)', fontweight='bold')
ax.set_xlabel('Importance Score')

save(fig, '04_model_evaluation')


# ── Confusion Matrix for Best Model ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, results['Gradient Boosting']['pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Non-Default', 'Default'])
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title('Confusion Matrix — Gradient Boosting', fontsize=13, fontweight='bold')
save(fig, '05_confusion_matrix')


# ════════════════════════════════════════════════════════════════════════════
# 6. PORTFOLIO SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════════════════
print("\n══ Portfolio Summary ══")

total_loans     = len(df)
total_exposure  = df['loan_amount'].sum()
default_rate    = df['is_default'].mean() * 100
avg_credit_score = df['credit_score'].mean()
avg_interest    = df['interest_rate'].mean()
expected_loss   = df[df['is_default'] == 1]['loan_amount'].sum()

summary = pd.DataFrame({
    'Metric': ['Total Loans', 'Total Exposure ($M)', 'Default Rate (%)',
               'Avg Credit Score', 'Avg Interest Rate (%)',
               'Expected Loss ($M)', 'Best Model AUC'],
    'Value': [f'{total_loans:,}',
              f'${total_exposure/1e6:.1f}M',
              f'{default_rate:.2f}%',
              f'{avg_credit_score:.0f}',
              f'{avg_interest:.2f}%',
              f'${expected_loss/1e6:.1f}M',
              f'{max(aucs):.4f}']
})
print(summary.to_string(index=False))
summary.to_csv('../reports/portfolio_summary.csv', index=False)

print("\n✅ All analysis complete. Visuals saved to /visuals/")
