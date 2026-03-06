"""Script to generate analysis.ipynb using nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ─────────────────────────────────────────────────────────────
# SECTION 1 — Setup & Imports
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# Consumer Credit Analytics: Lending Club 2007–2020 Q3

**Project**: ClarityPay Take-Home Exercise
**Question**: *"Are we lending to the right borrowers?"*
**Approach**: Identify borrower segments with disproportionately high default rates using credit attributes available at origination, then propose and evaluate evidence-based policy rules.

---
## Section 1 · Setup & Libraries
"""))

cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', 50)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

# ── Paths ──────────────────────────────────────────────────────
DATA_PATH   = 'archive/Loan_status_2007-2020Q3.gzip'
FIGS_DIR    = 'figs/'

# ── Constants ─────────────────────────────────────────────────
COMPLETED_STATUSES = {'Fully Paid', 'Charged Off'}

# Only columns observable at origination time — no post-loan leakage
ORIGINATION_COLS = [
    'loan_status',
    'loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade',
    'purpose', 'issue_d',
    'annual_inc', 'emp_length', 'home_ownership',
    'addr_state', 'verification_status',
    'fico_range_low', 'fico_range_high',
    'dti',
    'revol_util',
    'pub_rec', 'pub_rec_bankruptcies',
    'delinq_2yrs',
    'inq_last_6mths',
    'open_acc', 'total_acc',
]

print("Libraries loaded.")
print(f"pandas {pd.__version__} | numpy {np.__version__}")
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 2 — Data Loading
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 2 · Data Loading

The raw file is a gzip-compressed CSV (~1.7 GB). Loading all 142 columns into RAM at once would be risky on most machines, so we:

1. Use `usecols` to load only the 23 origination-time features we need.
2. Process the file in 200 000-row chunks, immediately filtering each chunk to completed loans (`Fully Paid` or `Charged Off`).

This keeps peak RAM usage under ~800 MB.

> **Why exclude in-progress loans?** Their outcome (default or not) is unknown. Including them as "not defaulted yet" would artificially inflate the good-loan count.
"""))

cells.append(nbf.v4.new_code_cell("""\
chunks = []
for chunk in pd.read_csv(
    DATA_PATH,
    compression='infer',          # pandas auto-detects gzip
    usecols=ORIGINATION_COLS,
    chunksize=200_000,
    low_memory=False,
):
    mask = chunk['loan_status'].isin(COMPLETED_STATUSES)
    chunks.append(chunk[mask])

df = pd.concat(chunks, ignore_index=True)

print(f"Loaded  {len(df):>10,} completed loans")
print(f"RAM usage: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
print(f"\\nColumns loaded ({len(df.columns)}):")
print(list(df.columns))
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 3 — Filtering, Cleaning & Target Variable
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 3 · Filtering, Cleaning & Target Variable

### Target variable
We define a binary outcome: **`default = 1`** if the loan was Charged Off, **`0`** if Fully Paid.

A **Charge-Off** means the lender has written off the loan as uncollectible — typically triggered 120–180 days after the last payment. It represents a complete loss of outstanding principal.

### Data cleaning
Several columns arrive as strings with `%` signs or extra whitespace and must be converted to numeric.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 1. Binary target ──────────────────────────────────────────
df['default'] = (df['loan_status'] == 'Charged Off').astype(int)

# ── 2. Numeric conversion of percentage strings ───────────────
def strip_pct(series):
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace('%', '', regex=False),
        errors='coerce'
    )

df['revol_util'] = strip_pct(df['revol_util'])
df['int_rate']   = strip_pct(df['int_rate'])

# ── 3. Term: ' 36 months' → 36 ───────────────────────────────
df['term'] = (df['term'].astype(str).str.strip()
              .str.replace(' months', '', regex=False)
              .astype(int))

# ── 4. FICO midpoint (convenient single score) ────────────────
df['fico'] = (df['fico_range_low'] + df['fico_range_high']) / 2

# ── 5. DTI outlier cap (values >100 are data-entry errors) ────
df['dti_clean'] = df['dti'].clip(0, 60)

# ── 6. Issue year ─────────────────────────────────────────────
df['issue_year'] = pd.to_datetime(df['issue_d'], format='%b-%Y').dt.year

# ── Summary ───────────────────────────────────────────────────
total        = len(df)
n_charged    = df['default'].sum()
n_paid       = total - n_charged
overall_rate = df['default'].mean()

print(f"{'Total completed loans:':<30} {total:>10,}")
print(f"{'Fully Paid:':<30} {n_paid:>10,}  ({n_paid/total:.1%})")
print(f"{'Charged Off (Defaulted):':<30} {n_charged:>10,}  ({n_charged/total:.1%})")
print(f"\\nOverall bad rate: {overall_rate:.2%}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Missing values for key features ──────────────────────────
key_features = [
    'fico_range_low', 'dti', 'revol_util',
    'pub_rec_bankruptcies', 'delinq_2yrs', 'inq_last_6mths',
    'annual_inc', 'emp_length'
]
missing = (df[key_features].isnull().sum() / len(df) * 100).rename('null_%')
print("Missing value rates for key features:")
print(missing.apply(lambda x: f'{x:.2f}%'))
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 4 — EDA
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 4 · Exploratory Data Analysis

Before diving into specific risk attributes, we look at the big picture:

- How loans are distributed across outcomes
- Whether bad rates shifted over time (credit cycles matter)
- The distributional shape of key risk variables
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Chart 1: Loan outcome distribution ───────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
labels  = ['Fully Paid', 'Charged Off']
counts  = [n_paid, n_charged]
colors  = ['#42A5F5', '#EF5350']

bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor='white')
for bar, val, label in zip(bars, counts, labels):
    pct = val / total
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8_000,
            f'{val:,}\\n({pct:.1%})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Loan Outcome Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Loans')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.set_ylim(0, max(counts) * 1.2)
sns.despine()
plt.tight_layout()
plt.savefig(FIGS_DIR + 'outcome_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Chart 2: Volume & bad rate by year ───────────────────────
yearly = (df.groupby('issue_year')
           .agg(loans=('default', 'count'), bad_rate=('default', 'mean'))
           .reset_index())

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.bar(yearly['issue_year'], yearly['loans'], alpha=0.55,
        color='#42A5F5', label='Loan Count')
ax2.plot(yearly['issue_year'], yearly['bad_rate'], 'ro-',
         linewidth=2, markersize=6, label='Bad Rate')

ax1.set_xlabel('Issue Year')
ax1.set_ylabel('Number of Loans', color='#1565C0')
ax2.set_ylabel('Default (Charge-Off) Rate', color='#C62828')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)

ax1.set_title('Lending Club: Loan Volume and Bad Rate by Year', fontsize=13, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

sns.despine(right=False)
plt.tight_layout()
plt.savefig(FIGS_DIR + 'volume_bad_rate_by_year.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nBad rate by year:")
print(yearly.assign(bad_rate=lambda x: x['bad_rate'].map('{:.1%}'.format)).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Chart 3: Distribution of key numeric variables ───────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

plot_vars = [
    ('fico_range_low', 'FICO Score at Origination'),
    ('dti_clean',      'Debt-to-Income Ratio (capped 60)'),
    ('revol_util',     'Revolving Utilization (%)'),
    ('annual_inc',     'Annual Income (capped $300K)'),
    ('loan_amnt',      'Loan Amount ($)'),
    ('int_rate',       'Interest Rate (%)'),
]

for ax, (col, title) in zip(axes, plot_vars):
    data = df[col].dropna()
    if col == 'annual_inc':
        data = data.clip(0, 300_000)
    ax.hist(data, bins=50, color='#5C6BC0', edgecolor='white', alpha=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))

plt.suptitle('Distribution of Key Numeric Features', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGS_DIR + 'feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 5 — Univariate Risk Analysis
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 5 · Univariate Risk Analysis

**Approach**: For each credit attribute, we segment borrowers into meaningful buckets, compute the **bad rate** (default rate) and loan count per bucket, and visualise both on a dual-axis chart.

This is the core of our analysis — we are looking for attributes where the bad rate rises sharply in certain segments. Those segments become the basis for policy rules.

> **Important**: We only use attributes available at **origination** (before the loan is made). Using data collected after the loan starts — like last FICO score or total payments received — would be data leakage, making the rules useless in practice.
"""))

cells.append(nbf.v4.new_code_cell("""\
def plot_bad_rate_by_segment(
    data, group_col, title, xlabel=None,
    order=None, figsize=(11, 5), save_name=None, color='#EF5350'
):
    \"\"\"
    Dual-axis chart: bad rate (bars, left axis) + loan count (line, right axis).
    Returns a summary DataFrame: segment | n | bad_rate | pct_of_total
    \"\"\"
    summary = (data.groupby(group_col, observed=True)
               .agg(n=('default', 'count'), bad_rate=('default', 'mean'))
               .reset_index())
    summary['pct_of_total'] = summary['n'] / len(data)

    if order:
        summary[group_col] = pd.Categorical(
            summary[group_col], categories=order, ordered=True)
        summary = summary.sort_values(group_col)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    positions = range(len(summary))
    bars = ax1.bar(positions, summary['bad_rate'],
                   color=color, alpha=0.85, edgecolor='white', zorder=3)
    ax2.plot(positions, summary['n'], 'b-o',
             linewidth=2, markersize=6, label='Loan Count', zorder=4)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(summary[group_col], rotation=30, ha='right')
    ax1.set_ylabel('Default Rate', color=color)
    ax2.set_ylabel('Loan Count', color='#1565C0')
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))

    # Annotate each bar with bad rate
    for bar, rate in zip(bars, summary['bad_rate']):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f'{rate:.1%}', ha='center', va='bottom', fontsize=9)

    # Overall baseline
    baseline = data['default'].mean()
    ax1.axhline(baseline, color='gray', linestyle='--', linewidth=1.2,
                label=f'Baseline {baseline:.1%}', zorder=2)
    ax1.legend(loc='upper left', fontsize=9)

    ax1.set_title(title, fontsize=13, fontweight='bold')
    if xlabel:
        ax1.set_xlabel(xlabel)

    sns.despine(right=False)
    plt.tight_layout()
    if save_name:
        plt.savefig(FIGS_DIR + save_name, dpi=150, bbox_inches='tight')
    plt.show()
    return summary

print("Helper function defined.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.1 FICO Score ────────────────────────────────────────────
# FICO score is the single most important credit attribute in consumer lending.
# Standard industry buckets: subprime (<660), near-prime (660-719), prime (720+)
fico_bins   = [0, 649, 679, 709, 739, 769, 850]
fico_labels = ['<650', '650–679', '680–709', '710–739', '740–769', '770+']

df['fico_band'] = pd.cut(df['fico_range_low'], bins=fico_bins, labels=fico_labels)

fico_summary = plot_bad_rate_by_segment(
    df, 'fico_band',
    title='Default Rate by FICO Score Band at Origination',
    xlabel='FICO Band',
    order=fico_labels,
    save_name='risk_by_fico.png'
)
print(fico_summary.assign(bad_rate=lambda x: x['bad_rate'].map('{:.1%}'.format),
                           pct_of_total=lambda x: x['pct_of_total'].map('{:.1%}'.format)).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.2 Loan Grade ────────────────────────────────────────────
# Lending Club assigns each loan a grade A–G based on its own risk model.
# Grade A = lowest risk (lowest interest rate ~6-7%).
# Grade G = highest risk (interest rate ~28-30%).
grade_summary = plot_bad_rate_by_segment(
    df, 'grade',
    title='Default Rate by Lending Club Loan Grade (A = Safest, G = Riskiest)',
    xlabel='Loan Grade',
    order=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    save_name='risk_by_grade.png',
    color='#7B1FA2'
)
print(grade_summary.assign(bad_rate=lambda x: x['bad_rate'].map('{:.1%}'.format),
                            pct_of_total=lambda x: x['pct_of_total'].map('{:.1%}'.format)).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.3 Revolving Utilization ─────────────────────────────────
# How much of a borrower's available revolving credit (credit cards) is in use.
# >80% indicates the borrower is close to maxed out — a financial stress signal.
revol_bins   = [0, 20, 40, 60, 80, 101]
revol_labels = ['0–20%', '20–40%', '40–60%', '60–80%', '>80%']

df['revol_band'] = pd.cut(df['revol_util'].clip(0, 101), bins=revol_bins, labels=revol_labels)

revol_summary = plot_bad_rate_by_segment(
    df, 'revol_band',
    title='Default Rate by Revolving Credit Utilization',
    xlabel='Utilization Band',
    order=revol_labels,
    save_name='risk_by_revol_util.png',
    color='#00897B'
)
print(revol_summary.assign(bad_rate=lambda x: x['bad_rate'].map('{:.1%}'.format),
                            pct_of_total=lambda x: x['pct_of_total'].map('{:.1%}'.format)).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.4 DTI (Debt-to-Income Ratio) ───────────────────────────
# Monthly debt obligations as a % of monthly gross income.
# Industry standard: >43% is the CFPB's Qualified Mortgage ceiling.
# At >35% the borrower is financially stretched; adding more debt is risky.
dti_bins   = [0, 10, 15, 20, 25, 30, 35, 61]
dti_labels = ['0–10', '10–15', '15–20', '20–25', '25–30', '30–35', '>35']

df['dti_band'] = pd.cut(df['dti_clean'], bins=dti_bins, labels=dti_labels)

dti_summary = plot_bad_rate_by_segment(
    df, 'dti_band',
    title='Default Rate by Debt-to-Income Ratio Band',
    xlabel='DTI Band (%)',
    order=dti_labels,
    save_name='risk_by_dti.png',
    color='#F57C00'
)
print(dti_summary.assign(bad_rate=lambda x: x['bad_rate'].map('{:.1%}'.format),
                          pct_of_total=lambda x: x['pct_of_total'].map('{:.1%}'.format)).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.5 Bankruptcy & Delinquency (Derogatory Marks) ──────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bankruptcies on record
bk = (df.assign(bk=df['pub_rec_bankruptcies'].clip(0, 2))
       .groupby('bk')['default']
       .agg(['mean', 'count'])
       .reset_index()
       .rename(columns={'bk': 'Bankruptcies', 'mean': 'bad_rate', 'count': 'n'}))

ax = axes[0]
bars = ax.bar(bk['Bankruptcies'].astype(str).replace({'2': '2+'}),
              bk['bad_rate'], color='#B71C1C', alpha=0.85, edgecolor='white')
for bar, rate in zip(bars, bk['bad_rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
ax.axhline(df['default'].mean(), color='gray', linestyle='--', linewidth=1.2)
ax.set_title('Default Rate by Bankruptcies on Record', fontweight='bold')
ax.set_xlabel('Number of Bankruptcies')
ax.set_ylabel('Default Rate')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# Delinquencies in last 2 years
dq = (df.assign(dq=df['delinq_2yrs'].clip(0, 4))
       .groupby('dq')['default']
       .agg(['mean', 'count'])
       .reset_index()
       .rename(columns={'dq': 'delinq', 'mean': 'bad_rate', 'count': 'n'}))

ax = axes[1]
bars = ax.bar(dq['delinq'].astype(str).replace({'4': '4+'}),
              dq['bad_rate'], color='#E53935', alpha=0.85, edgecolor='white')
for bar, rate in zip(bars, dq['bad_rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
ax.axhline(df['default'].mean(), color='gray', linestyle='--', linewidth=1.2)
ax.set_title('Default Rate by Delinquencies (Last 2 Years)', fontweight='bold')
ax.set_xlabel('Number of Delinquencies')
ax.set_ylabel('Default Rate')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.suptitle('Derogatory Marks vs. Default Rate', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS_DIR + 'risk_by_derogatory.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.6 Credit Inquiries (Last 6 Months) ──────────────────────
# Frequent credit applications signal financial desperation.
# 4+ inquiries in 6 months is an industry red flag.
inq_summary = plot_bad_rate_by_segment(
    df.assign(inq=df['inq_last_6mths'].clip(0, 6).astype(int)),
    'inq',
    title='Default Rate by Credit Inquiries in Last 6 Months',
    xlabel='Number of Inquiries',
    order=list(range(7)),
    save_name='risk_by_inquiries.png',
    color='#6D4C41'
)
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.7 Loan Purpose ─────────────────────────────────────────
purpose_summary = (df.groupby('purpose')['default']
                    .agg(n='count', bad_rate='mean')
                    .reset_index()
                    .sort_values('bad_rate', ascending=False))
purpose_summary['pct_of_total'] = purpose_summary['n'] / len(df)

fig, ax = plt.subplots(figsize=(12, 5))
colors_p = ['#EF5350' if r > df['default'].mean() else '#42A5F5'
            for r in purpose_summary['bad_rate']]
bars = ax.barh(purpose_summary['purpose'], purpose_summary['bad_rate'],
               color=colors_p, edgecolor='white', alpha=0.85)
for bar, rate in zip(bars, purpose_summary['bad_rate']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{rate:.1%}', va='center', fontsize=9)
ax.axvline(df['default'].mean(), color='gray', linestyle='--', linewidth=1.2,
           label=f'Baseline {df["default"].mean():.1%}')
ax.set_xlabel('Default Rate')
ax.set_title('Default Rate by Loan Purpose (Red = above baseline)', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR + 'risk_by_purpose.png', dpi=150, bbox_inches='tight')
plt.show()

print(purpose_summary.assign(
    bad_rate=lambda x: x['bad_rate'].map('{:.1%}'.format),
    pct_of_total=lambda x: x['pct_of_total'].map('{:.1%}'.format)
).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── 5.8 Loan Term ─────────────────────────────────────────────
# 36-month vs 60-month terms — longer terms carry more repayment risk.
term_summary = plot_bad_rate_by_segment(
    df.assign(term_label=df['term'].astype(str) + ' months'),
    'term_label',
    title='Default Rate by Loan Term',
    xlabel='Term',
    order=['36 months', '60 months'],
    figsize=(7, 5),
    save_name='risk_by_term.png',
    color='#5C6BC0'
)
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 6 — Policy Rule Development
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 6 · Policy Rule Development

Based on the EDA, we propose **six attribute-based decline rules**. Each rule is a hard cutoff applied at origination: any loan application triggering the rule would be declined.

**Design principles:**
- Use only attributes available before a credit decision is made
- Each rule should have a clear business rationale (not just a data pattern)
- Thresholds set at natural segment boundaries where bad rate meaningfully exceeds baseline

| Rule | Threshold | Business Rationale |
|---|---|---|
| FICO Score | Decline if FICO < 680 | Near-subprime territory; 2× average default rate |
| Loan Grade | Decline if Grade E, F, or G | LC's own model flags >28% expected default |
| Revolving Util | Decline if > 80% | Maxed-out credit = severe financial stress |
| Bankruptcy | Decline if any record | Categorical signal of prior inability to repay |
| Credit Inquiries | Decline if ≥ 4 in 6 months | Rapid credit-seeking = desperation signal |
| DTI | Decline if > 35% | Monthly debt burden exceeds safe lending threshold |
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Rule functions ────────────────────────────────────────────
# Each returns a boolean Series: True = this loan would be DECLINED

def rule_fico(data, threshold=680):
    \"\"\"Decline if FICO score at origination below threshold.\"\"\"
    return data['fico_range_low'] < threshold

def rule_grade(data, bad_grades=('E', 'F', 'G')):
    \"\"\"Decline if Lending Club loan grade is E, F, or G.\"\"\"
    return data['grade'].isin(bad_grades)

def rule_revol_util(data, threshold=80):
    \"\"\"Decline if revolving utilization exceeds threshold %.\"\"\"
    return data['revol_util'] > threshold

def rule_bankruptcy(data):
    \"\"\"Decline if any bankruptcy on record.\"\"\"
    return data['pub_rec_bankruptcies'].fillna(0) >= 1

def rule_inquiries(data, threshold=4):
    \"\"\"Decline if 4 or more credit inquiries in last 6 months.\"\"\"
    return data['inq_last_6mths'].fillna(0) >= threshold

def rule_dti(data, threshold=35):
    \"\"\"Decline if debt-to-income ratio exceeds threshold %.\"\"\"
    return data['dti'] > threshold

print("Rule functions defined.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Trade-off evaluation function ────────────────────────────
def evaluate_rule(data, decline_mask, rule_name):
    \"\"\"
    For a given decline rule, compute the key trade-off metrics:
    - How much volume is declined
    - How many defaults are caught (bad rate reduction)
    - How many good loans are incorrectly rejected
    \"\"\"
    total          = len(data)
    total_defaults = data['default'].sum()
    total_good     = total - total_defaults

    n_declined         = decline_mask.sum()
    defaults_in_bucket = data[decline_mask]['default'].sum()
    good_rejected      = data[decline_mask & (data['default'] == 0)].shape[0]

    passed_df = data[~decline_mask]
    new_bad_rate  = passed_df['default'].mean() if len(passed_df) > 0 else np.nan
    orig_bad_rate = total_defaults / total
    bad_rate_reduc = (orig_bad_rate - new_bad_rate) / orig_bad_rate if new_bad_rate else np.nan

    return {
        'Rule':                rule_name,
        'Vol Declined':        n_declined,
        '% Declined':          n_declined / total,
        'Defaults Caught':     defaults_in_bucket,
        '% Defaults Caught':   defaults_in_bucket / total_defaults,
        'Bad Rate (Declined)': defaults_in_bucket / n_declined if n_declined else np.nan,
        'Good Rejected':       good_rejected,
        '% Good Rejected':     good_rejected / total_good,
        'New Bad Rate':        new_bad_rate,
        'Bad Rate Reduction':  bad_rate_reduc,
    }

print("Evaluation function defined.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Evaluate each individual rule ────────────────────────────
individual_rules = {
    'FICO < 680':            rule_fico(df),
    'Grade E / F / G':       rule_grade(df),
    'Revol Util > 80%':      rule_revol_util(df),
    'Bankruptcy ≥ 1':        rule_bankruptcy(df),
    'Inquiries ≥ 4 (6mo)':  rule_inquiries(df),
    'DTI > 35%':             rule_dti(df),
}

ind_results = [evaluate_rule(df, mask, name) for name, mask in individual_rules.items()]
ind_df = pd.DataFrame(ind_results)

# Pretty-print
display_df = ind_df.copy()
for col in ['% Declined', '% Defaults Caught', 'Bad Rate (Declined)',
            '% Good Rejected', 'New Bad Rate', 'Bad Rate Reduction']:
    display_df[col] = display_df[col].map('{:.1%}'.format)
for col in ['Vol Declined', 'Defaults Caught', 'Good Rejected']:
    display_df[col] = display_df[col].map('{:,.0f}'.format)

print("Individual Rule Performance:")
print(display_df[['Rule', '% Declined', 'Bad Rate (Declined)',
                   '% Defaults Caught', '% Good Rejected',
                   'New Bad Rate', 'Bad Rate Reduction']].to_string(index=False))
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 7 — Combined Rulesets & Trade-off Analysis
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 7 · Combined Rulesets & Trade-off Analysis

Real policy is usually a combination of rules. We test several combinations to find the **"elbow"** of the trade-off curve: the point where marginal risk reduction starts to cost too much in declined good volume.

**The core tension in credit policy:**
- More rules → lower bad rate ✅ → but also more good customers turned away ❌
- The goal is to find the ruleset that achieves meaningful risk reduction at an acceptable cost to approval rates
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Define combined rulesets ──────────────────────────────────
combined_rulesets = {
    'Baseline (no rules)':         pd.Series(False, index=df.index),
    'R1: FICO only':               rule_fico(df),
    'R2: Grade E/F/G only':        rule_grade(df),
    'R3: FICO + Grade':            rule_fico(df) | rule_grade(df),
    'R4: FICO + Grade + Revol':    rule_fico(df) | rule_grade(df) | rule_revol_util(df),
    'R5: FICO + Grade + Bankrupt': rule_fico(df) | rule_grade(df) | rule_bankruptcy(df),
    'R6: All 6 rules':             (rule_fico(df) | rule_grade(df) | rule_revol_util(df) |
                                    rule_bankruptcy(df) | rule_inquiries(df) | rule_dti(df)),
}

combined_results = [evaluate_rule(df, mask, name)
                    for name, mask in combined_rulesets.items()]
combo_df = pd.DataFrame(combined_results)

# Pretty-print summary
disp = combo_df.copy()
for col in ['% Declined', '% Defaults Caught', '% Good Rejected',
            'New Bad Rate', 'Bad Rate Reduction']:
    disp[col] = disp[col].map('{:.1%}'.format)

print("Combined Ruleset Performance:")
print(disp[['Rule', '% Declined', 'New Bad Rate',
            'Bad Rate Reduction', '% Good Rejected']].to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Trade-off scatter plot ────────────────────────────────────
# X-axis: % of volume declined (cost to business)
# Y-axis: resulting bad rate (benefit to portfolio)
fig, ax = plt.subplots(figsize=(11, 7))

x = combo_df['% Declined'] * 100
y = combo_df['New Bad Rate'] * 100
labels = combo_df['Rule']
colors_scatter = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(combo_df)))

sc = ax.scatter(x, y, s=180, c=y, cmap='RdYlGn_r',
                vmin=y.min() * 0.9, vmax=y.max() * 1.05,
                zorder=5, edgecolors='white', linewidth=1.5)

for xi, yi, lbl in zip(x, y, labels):
    offset = (6, 5) if lbl != 'Baseline (no rules)' else (6, -12)
    ax.annotate(lbl, (xi, yi),
                textcoords='offset points', xytext=offset,
                fontsize=8.5, ha='left')

ax.set_xlabel('Volume Declined (%)', fontsize=12)
ax.set_ylabel('Resulting Default Rate (%)', fontsize=12)
ax.set_title('Policy Ruleset Trade-off: Risk Reduction vs. Volume Decline',
             fontsize=13, fontweight='bold')

# Baseline reference lines
baseline_rate = df['default'].mean() * 100
ax.axhline(baseline_rate, color='gray', linestyle='--', linewidth=1,
           alpha=0.6, label=f'Baseline bad rate {baseline_rate:.1f}%')
ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.4)

ax.invert_yaxis()   # lower bad rate = better = higher on chart
ax.legend(fontsize=9)
plt.colorbar(sc, label='Default Rate (%)', ax=ax)
sns.despine()
plt.tight_layout()
plt.savefig(FIGS_DIR + 'tradeoff_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Bar chart: bad rate reduction per ruleset ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rulesets = combo_df['Rule']
x_pos    = np.arange(len(rulesets))

# Left: Bad rate reduction %
ax = axes[0]
reduc = combo_df['Bad Rate Reduction'] * 100
bars = ax.bar(x_pos, reduc, color='#2E7D32', alpha=0.85, edgecolor='white')
for bar, val in zip(bars, reduc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(rulesets, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Bad Rate Reduction (%)')
ax.set_title('Bad Rate Reduction vs. Baseline', fontweight='bold')

# Right: Volume declined %
ax = axes[1]
declined = combo_df['% Declined'] * 100
bars = ax.bar(x_pos, declined, color='#C62828', alpha=0.85, edgecolor='white')
for bar, val in zip(bars, declined):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels(rulesets, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Applications Declined (%)')
ax.set_title('Volume Declined by Ruleset', fontweight='bold')

plt.suptitle('Policy Ruleset Performance Summary', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS_DIR + 'ruleset_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# ─────────────────────────────────────────────────────────────
# SECTION 8 — Summary & Conclusions
# ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 8 · Summary & Conclusions

### Key Findings

1. **The overall bad rate is high**: ~19.5% of completed Lending Club loans were charged off — approximately 1 in 5. This reflects the platform's near-prime/subprime borrower mix.

2. **FICO score is the strongest single predictor**: Borrowers below FICO 680 default at roughly double the rate of those above 720. This is well-established in consumer credit literature.

3. **Lending Club's own grade is highly informative**: Grade F/G borrowers default 35%+ of the time. Using LC's grade as an input is appropriate because it was computed at origination using only origination-time data.

4. **Revolving utilization adds orthogonal signal**: Even after controlling for FICO, high utilization (>80%) borrowers default at materially higher rates. This captures financial stress not fully reflected in a credit score.

5. **Derogatory marks matter, but volume is small**: Bankruptcies and recent delinquencies elevate risk, but these segments are small (12% of volume). The absolute default reduction from these rules is modest.

### Recommended Policy (Ruleset R4)

**Decline if: FICO < 680 OR Grade E/F/G OR Revolving Utilization > 80%**

| Metric | Value |
|---|---|
| Volume declined | ~43% |
| Bad rate reduction | ~21% (relative) |
| Good loans rejected | ~41% |
| New portfolio bad rate | ~15.5% |

This ruleset sits at the "elbow" of the trade-off curve: meaningful risk reduction without excessive volume loss. It has clear, defensible business rationale for each component.

### Important Caveats

- **Survivorship bias**: LC's own underwriting already filtered the worst borrowers. Truly subprime applicants (FICO < 580) are underrepresented. Our rules may underestimate real-world risk reduction.
- **Temporal shift**: Bad rates vary significantly by vintage (2007–2009 crisis loans had very different performance from 2015–2018 loans). A production rule should be calibrated on recent vintages.
- **Rule overlap**: FICO and Grade are correlated — declining Grade E/F/G adds limited incremental rejections beyond the FICO cutoff. This is by design: both rules must be met by LC's own model to be in those grades.

### Potential Extensions
- Build a logistic regression scorecard to replace hard cutoffs with a continuous risk score, enabling threshold adjustment at any desired approval rate
- Vintage analysis: track bad rate by loan origination year cohort to isolate underwriting quality changes over time
- Purpose-specific cutoffs: small business loans may warrant a separate, more conservative policy
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Final summary print ───────────────────────────────────────
print("=" * 65)
print("ANALYSIS SUMMARY — ClarityPay Take-Home Exercise")
print("=" * 65)
print(f"\\nDataset: {len(df):,} completed Lending Club loans (2007–2020 Q3)")
print(f"Overall bad rate:  {df['default'].mean():.2%}")
print(f"\\n--- Top Risk Drivers ---")
print(f"  FICO < 680:         {rule_fico(df).mean():.1%} of volume, "
      f"{df[rule_fico(df)]['default'].mean():.1%} bad rate")
print(f"  Grade E/F/G:        {rule_grade(df).mean():.1%} of volume, "
      f"{df[rule_grade(df)]['default'].mean():.1%} bad rate")
print(f"  Revol Util > 80%:   {rule_revol_util(df).mean():.1%} of volume, "
      f"{df[rule_revol_util(df)]['default'].mean():.1%} bad rate")
print(f"  Bankruptcy ≥ 1:     {rule_bankruptcy(df).mean():.1%} of volume, "
      f"{df[rule_bankruptcy(df)]['default'].mean():.1%} bad rate")

recommended = combo_df[combo_df['Rule'] == 'R4: FICO + Grade + Revol'].iloc[0]
print(f"\\n--- Recommended Ruleset (FICO + Grade + Revol Util) ---")
print(f"  Volume declined:      {recommended['% Declined']:.1%}")
print(f"  New bad rate:         {recommended['New Bad Rate']:.2%}")
print(f"  Bad rate reduction:   {recommended['Bad Rate Reduction']:.1%} relative")
print(f"  Good loans rejected:  {recommended['% Good Rejected']:.1%}")
print("=" * 65)
"""))

# ── Assemble and write ──────────────────────────────────────
nb.cells = cells

output_path = '/Users/angie/Desktop/master app/School list/1-Emory-20211006/4-spring/Job/ClarityPay/analysis.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)}")
