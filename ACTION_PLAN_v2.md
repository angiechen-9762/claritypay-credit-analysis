# ClarityPay Take-Home: Improved Action Plan (v2)

## What Changed from v1

This plan builds on the v1 structure but improves it in several specific ways:

1. **Interaction/bivariate analysis added** — FICO × DTI heatmap and FICO × Revol Util heatmap to find the "worst pockets" where variables compound
2. **Swap set framing** — every rule is evaluated as a swap set with explicit good-to-bad ratio
3. **Rule comparison table** — a single summary table comparing all candidate rules and combinations side-by-side
4. **Efficient frontier chart** — scatter plot of (% good volume declined) vs (bad rate reduction) to visualize the trade-off curve
5. **Slide deck restructured** — follows the narrative arc: problem → drivers → segments → rules → trade-offs → next steps, with specific content per slide
6. **Vintage analysis as bonus** — cumulative default curves by origination year to show time-varying risk
7. **Dollar-denominated loss framing** — translate bad rates into estimated dollar losses for business impact

---

## Phase 1: Data Loading & Cleaning

### Cell 1 — Setup & Constants
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.ticker import FuncFormatter

DATA_PATH = 'archive/Loan_status_2007-2020Q3.gzip'
COMPLETED_STATUSES = {'Fully Paid', 'Charged Off'}

# Style
sns.set_theme(style='whitegrid', palette='muted')
COLOR_GOOD = '#4C72B0'   # blue for Fully Paid
COLOR_BAD  = '#C44E52'   # red for Charged Off
COLOR_NEUTRAL = '#8C8C8C'

plt.rcParams.update({
    'figure.figsize': (10, 5),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
```

### Cell 2 — Columns to Load
```python
# ONLY origination-time features — no post-loan data
ORIGINATION_COLS = [
    'loan_status', 'loan_amnt', 'funded_amnt', 'term', 'int_rate',
    'grade', 'sub_grade', 'purpose', 'issue_d',
    'annual_inc', 'emp_length', 'home_ownership',
    'addr_state', 'verification_status',
    'fico_range_low', 'fico_range_high',
    'dti', 'revol_util', 'revol_bal',
    'pub_rec', 'pub_rec_bankruptcies',
    'delinq_2yrs', 'inq_last_6mths',
    'open_acc', 'total_acc',
]
```

**Markdown commentary to include:**
> "We deliberately restrict to origination-time attributes to avoid data leakage. Post-origination fields like `total_pymnt`, `recoveries`, `last_fico_range_high`, `hardship_flag`, and `debt_settlement_flag` are excluded — using these would mean our rules rely on information unavailable at the point of the lending decision."

### Cell 3 — Chunked Loading
```python
chunks = []
for chunk in pd.read_csv(DATA_PATH, compression='infer',
                          usecols=ORIGINATION_COLS,
                          chunksize=200_000, low_memory=False):
    mask = chunk['loan_status'].isin(COMPLETED_STATUSES)
    chunks.append(chunk[mask])
df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} completed loans")
```

### Cell 4 — Cleaning & Target Variable
```python
# Target
df['default'] = (df['loan_status'] == 'Charged Off').astype(int)

# Clean string-encoded numerics
df['revol_util'] = df['revol_util'].astype(str).str.rstrip('%').replace('nan', np.nan).astype(float)
df['int_rate']   = df['int_rate'].astype(str).str.strip().str.rstrip('%').replace('nan', np.nan).astype(float)
df['term']       = df['term'].astype(str).str.strip().str.replace(' months','').astype(float).astype('Int64')

# Cap DTI outliers at 60 (values up to 999 exist)
df['dti'] = df['dti'].clip(upper=60)

# FICO midpoint for easier analysis
df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2

# Issue date
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
df['issue_year'] = df['issue_d'].dt.year

# Estimated loss per charged-off loan (simple: assume lose remaining principal)
# Use loan_amnt as proxy for loss exposure
df['loss_exposure'] = df['loan_amnt']
```

### Cell 5 — Baseline Summary
```python
total = len(df)
n_bad = df['default'].sum()
n_good = total - n_bad
bad_rate = n_bad / total
total_exposure = df['loan_amnt'].sum()
total_loss = df.loc[df['default']==1, 'loan_amnt'].sum()

print(f"Total completed loans: {total:,}")
print(f"Fully Paid:  {n_good:,} ({n_good/total:.1%})")
print(f"Charged Off: {n_bad:,}  ({bad_rate:.1%})")
print(f"Total funded amount:   ${total_exposure:,.0f}")
print(f"Loss exposure (charged-off principal): ${total_loss:,.0f}")
```

**This is your baseline. Every rule you propose later gets measured against these numbers.**

---

## Phase 2: Exploratory Data Analysis

### Cell 6 — Missing Values
```python
missing = df[ORIGINATION_COLS].isnull().mean().sort_values(ascending=False)
missing[missing > 0].plot.barh(title='Missing Value Rates')
```

**Commentary:** Note which fields have significant missingness and how you'll handle them (drop rows for small %, note the limitation for larger %).

### Cell 7 — Reusable Plotting Function
```python
def plot_bad_rate(df, group_col, title, order=None, figsize=(10,5)):
    """Dual-axis chart: bars = loan volume, line = bad rate by segment."""
    agg = df.groupby(group_col, observed=True).agg(
        n_loans=('default','count'),
        bad_rate=('default','mean'),
        total_loss=('loan_amnt', lambda x: x[df.loc[x.index,'default']==1].sum())
    ).reset_index()
    if order is not None:
        agg[group_col] = pd.Categorical(agg[group_col], categories=order, ordered=True)
        agg = agg.sort_values(group_col)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    ax1.bar(range(len(agg)), agg['n_loans'], color=COLOR_NEUTRAL, alpha=0.5, label='Loan Volume')
    ax2.plot(range(len(agg)), agg['bad_rate'], color=COLOR_BAD, marker='o', linewidth=2, label='Bad Rate')
    ax2.axhline(df['default'].mean(), color=COLOR_BAD, linestyle='--', alpha=0.4, label='Portfolio Avg')

    ax1.set_xticks(range(len(agg)))
    ax1.set_xticklabels(agg[group_col], rotation=45, ha='right')
    ax1.set_ylabel('Loan Volume')
    ax2.set_ylabel('Bad Rate')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.set_title(title)
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    return agg
```

### Cell 8 — Volume & Bad Rate by Year (Vintage Overview)
```python
plot_bad_rate(df, 'issue_year', 'Loan Volume & Bad Rate by Origination Year')
```
**Commentary:** "This gives us a first look at how portfolio composition changed over time. Note the spike in bad rates during [year range] corresponding to [economic context]."

### Cells 9-16 — Univariate Risk Analysis

Create bins and run `plot_bad_rate` for each key attribute:

```python
# FICO bins
df['fico_bin'] = pd.cut(df['fico_range_low'],
    bins=[0, 649, 679, 709, 739, 769, 850],
    labels=['<650','650-679','680-709','710-739','740-769','770+'])

# DTI bins (already capped at 60)
df['dti_bin'] = pd.cut(df['dti'],
    bins=[-1, 10, 15, 20, 25, 30, 35, 60],
    labels=['0-10%','10-15%','15-20%','20-25%','25-30%','30-35%','>35%'])

# Revolving utilization bins
df['revol_util_bin'] = pd.cut(df['revol_util'],
    bins=[-1, 20, 40, 60, 80, 200],
    labels=['0-20%','20-40%','40-60%','60-80%','>80%'])

# Inquiries bins
df['inq_bin'] = df['inq_last_6mths'].clip(upper=5).astype(str)
df.loc[df['inq_last_6mths'] >= 5, 'inq_bin'] = '5+'

# Bankruptcies
df['bankrupt_bin'] = df['pub_rec_bankruptcies'].clip(upper=2).fillna(0).astype(int).astype(str)
df.loc[df['pub_rec_bankruptcies'] >= 2, 'bankrupt_bin'] = '2+'
```

Then for each:
```python
plot_bad_rate(df, 'fico_bin', 'Bad Rate by FICO Score', order=['<650','650-679','680-709','710-739','740-769','770+'])
plot_bad_rate(df, 'dti_bin', 'Bad Rate by DTI Ratio')
plot_bad_rate(df, 'revol_util_bin', 'Bad Rate by Revolving Utilization')
plot_bad_rate(df, 'grade', 'Bad Rate by Loan Grade', order=list('ABCDEFG'))
plot_bad_rate(df, 'purpose', 'Bad Rate by Loan Purpose')
plot_bad_rate(df, 'term', 'Bad Rate by Loan Term')
plot_bad_rate(df, 'inq_bin', 'Bad Rate by Recent Inquiries (Last 6 Months)')
plot_bad_rate(df, 'bankrupt_bin', 'Bad Rate by Bankruptcy Count')
```

**Commentary for each:** One sentence on the pattern ("monotonically increasing", "sharp jump at threshold X"), one sentence on business interpretation ("this aligns with the intuition that...").

---

## Phase 3: Interaction Analysis (NEW — not in v1)

This is where you find the "worst pockets" by combining variables.

### Cell 17 — FICO × DTI Heatmap
```python
# Create coarser bins for the heatmap (too many cells = unreadable)
df['fico_coarse'] = pd.cut(df['fico_range_low'],
    bins=[0, 659, 689, 719, 850],
    labels=['<660','660-689','690-719','720+'])
df['dti_coarse'] = pd.cut(df['dti'],
    bins=[-1, 15, 25, 35, 60],
    labels=['0-15%','15-25%','25-35%','>35%'])

heatmap_data = df.pivot_table(values='default', index='fico_coarse', columns='dti_coarse', aggfunc='mean')
heatmap_count = df.pivot_table(values='default', index='fico_coarse', columns='dti_coarse', aggfunc='count')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bad rate heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='YlOrRd', ax=axes[0],
            vmin=0.05, vmax=0.40)
axes[0].set_title('Bad Rate: FICO × DTI')

# Volume heatmap (so we know which cells matter)
sns.heatmap(heatmap_count, annot=True, fmt=',.0f', cmap='Blues', ax=axes[1])
axes[1].set_title('Loan Volume: FICO × DTI')

plt.tight_layout()
```

**Commentary:** "The top-left cell (low FICO, high DTI) shows a bad rate of ~X%, nearly Yx the portfolio average, and contains Z loans — a meaningful volume. This confirms that FICO and DTI compound: borrowers who are both credit-impaired and over-leveraged are the highest-risk segment."

### Cell 18 — FICO × Revolving Utilization Heatmap
```python
df['revol_coarse'] = pd.cut(df['revol_util'],
    bins=[-1, 40, 60, 80, 200],
    labels=['0-40%','40-60%','60-80%','>80%'])

heatmap2 = df.pivot_table(values='default', index='fico_coarse', columns='revol_coarse', aggfunc='mean')

sns.heatmap(heatmap2, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0.05, vmax=0.40)
plt.title('Bad Rate: FICO × Revolving Utilization')
plt.tight_layout()
```

### Cell 19 — Purpose × Grade Heatmap (Optional)
```python
# Focus on top-volume purposes
top_purposes = df['purpose'].value_counts().head(6).index
heatmap3 = (df[df['purpose'].isin(top_purposes)]
            .pivot_table(values='default', index='purpose', columns='grade', aggfunc='mean'))

sns.heatmap(heatmap3, annot=True, fmt='.1%', cmap='YlOrRd')
plt.title('Bad Rate: Purpose × Grade')
```

---

## Phase 4: Policy Rules & Swap Set Analysis

### Cell 20 — Define Rules
```python
# Each rule returns a boolean mask: True = WOULD BE DECLINED
RULES = {
    'FICO < 660':           lambda df: df['fico_range_low'] < 660,
    'FICO < 680':           lambda df: df['fico_range_low'] < 680,
    'DTI > 35%':            lambda df: df['dti'] > 35,
    'Revol Util > 80%':     lambda df: df['revol_util'] > 80,
    'Bankruptcy >= 1':      lambda df: df['pub_rec_bankruptcies'] >= 1,
    'Inquiries >= 4':       lambda df: df['inq_last_6mths'] >= 4,
    'Grade E/F/G':          lambda df: df['grade'].isin(['E','F','G']),
    '60-month term':        lambda df: df['term'] == 60,
}
```

### Cell 21 — Swap Set Evaluation Function (IMPROVED)
```python
def evaluate_swap_set(df, mask, rule_name):
    """
    Evaluate a policy rule by analyzing its swap set.
    Returns a dict with all key trade-off metrics.
    """
    total = len(df)
    total_bads = df['default'].sum()
    total_goods = total - total_bads
    baseline_bad_rate = total_bads / total

    declined = df[mask]
    approved = df[~mask]

    n_declined = len(declined)
    n_declined_bad = declined['default'].sum()
    n_declined_good = n_declined - n_declined_bad

    n_remaining = len(approved)
    n_remaining_bad = approved['default'].sum()
    new_bad_rate = n_remaining_bad / n_remaining if n_remaining > 0 else 0

    # Dollar impact
    loss_avoided = declined.loc[declined['default']==1, 'loan_amnt'].sum()
    good_revenue_lost = declined.loc[declined['default']==0, 'loan_amnt'].sum()

    return {
        'Rule': rule_name,
        'Swap Set Size': n_declined,
        'Swap Set %': n_declined / total,
        'Bad Rate in Swap Set': n_declined_bad / n_declined if n_declined > 0 else 0,
        'Bads Caught': n_declined_bad,
        'Bads Caught %': n_declined_bad / total_bads,
        'Goods Lost': n_declined_good,
        'Goods Lost %': n_declined_good / total_goods,
        'Good-to-Bad Ratio': n_declined_good / n_declined_bad if n_declined_bad > 0 else np.inf,
        'New Bad Rate': new_bad_rate,
        'Bad Rate Reduction (pp)': (baseline_bad_rate - new_bad_rate) * 100,
        'Bad Rate Reduction (rel%)': (baseline_bad_rate - new_bad_rate) / baseline_bad_rate * 100,
        'Loss Avoided ($)': loss_avoided,
        'Good Revenue Lost ($)': good_revenue_lost,
    }
```

### Cell 22 — Evaluate Individual Rules
```python
results = []
for name, rule_fn in RULES.items():
    mask = rule_fn(df).fillna(False)
    results.append(evaluate_swap_set(df, mask, name))

results_df = pd.DataFrame(results)

# Format for display
display_cols = ['Rule', 'Swap Set Size', 'Swap Set %', 'Bad Rate in Swap Set',
                'Bads Caught %', 'Goods Lost %', 'Good-to-Bad Ratio',
                'New Bad Rate', 'Bad Rate Reduction (pp)']
results_df[display_cols].style.format({
    'Swap Set %': '{:.1%}', 'Bad Rate in Swap Set': '{:.1%}',
    'Bads Caught %': '{:.1%}', 'Goods Lost %': '{:.1%}',
    'Good-to-Bad Ratio': '{:.1f}:1', 'New Bad Rate': '{:.1%}',
    'Bad Rate Reduction (pp)': '{:+.2f}pp', 'Swap Set Size': '{:,.0f}'
})
```

### Cell 23 — Evaluate Rule COMBINATIONS
```python
# Define candidate rulesets (combinations)
RULESETS = {
    'A: FICO<680 only':
        lambda df: RULES['FICO < 680'](df),
    'B: FICO<680 + Revol>80%':
        lambda df: RULES['FICO < 680'](df) | RULES['Revol Util > 80%'](df),
    'C: FICO<680 + Revol>80% + DTI>35%':
        lambda df: RULES['FICO < 680'](df) | RULES['Revol Util > 80%'](df) | RULES['DTI > 35%'](df),
    'D: FICO<660 + Grade E/F/G + Revol>80%':
        lambda df: RULES['FICO < 660'](df) | RULES['Grade E/F/G'](df) | RULES['Revol Util > 80%'](df),
    'E: FICO<660 + Bankruptcy>=1 + Revol>80%':
        lambda df: RULES['FICO < 660'](df) | RULES['Bankruptcy >= 1'](df) | RULES['Revol Util > 80%'](df),
    'F: All 6 rules (aggressive)':
        lambda df: (RULES['FICO < 680'](df) | RULES['DTI > 35%'](df) |
                    RULES['Revol Util > 80%'](df) | RULES['Bankruptcy >= 1'](df) |
                    RULES['Inquiries >= 4'](df) | RULES['Grade E/F/G'](df)),
}

combo_results = []
for name, ruleset_fn in RULESETS.items():
    mask = ruleset_fn(df).fillna(False)
    combo_results.append(evaluate_swap_set(df, mask, name))

combo_df = pd.DataFrame(combo_results)
```

### Cell 24 — Rule Comparison Table (THE KEY OUTPUT)
```python
# Combine individual + combo results into one comparison table
all_results = pd.concat([results_df, combo_df], ignore_index=True)

# Add baseline row at top
baseline_row = {
    'Rule': 'BASELINE (no rules)',
    'Swap Set Size': 0, 'Swap Set %': 0,
    'Bad Rate in Swap Set': np.nan,
    'Bads Caught %': 0, 'Goods Lost %': 0,
    'Good-to-Bad Ratio': np.nan,
    'New Bad Rate': df['default'].mean(),
    'Bad Rate Reduction (pp)': 0,
}
all_results = pd.concat([pd.DataFrame([baseline_row]), all_results], ignore_index=True)
```

**Commentary:** "The comparison table shows that Ruleset D (FICO<660 + Grade E/F/G + Revol>80%) achieves a strong bad rate reduction with a relatively efficient good-to-bad ratio of X:1. More aggressive rulesets (F) catch more defaults but at steeply increasing cost to good volume."

---

## Phase 5: Trade-Off Visualization

### Cell 25 — Efficient Frontier Chart (NEW)
```python
fig, ax = plt.subplots(figsize=(10, 7))

# Plot individual rules as circles
for _, row in results_df.iterrows():
    ax.scatter(row['Goods Lost %'] * 100, row['Bad Rate Reduction (pp)'],
               s=100, color=COLOR_NEUTRAL, zorder=3, edgecolors='black')
    ax.annotate(row['Rule'], (row['Goods Lost %']*100, row['Bad Rate Reduction (pp)']),
                textcoords="offset points", xytext=(8, 4), fontsize=8)

# Plot combo rulesets as diamonds (highlighted)
for _, row in combo_df.iterrows():
    ax.scatter(row['Goods Lost %'] * 100, row['Bad Rate Reduction (pp)'],
               s=150, color=COLOR_BAD, marker='D', zorder=4, edgecolors='black')
    ax.annotate(row['Rule'], (row['Goods Lost %']*100, row['Bad Rate Reduction (pp)']),
                textcoords="offset points", xytext=(8, 4), fontsize=8, fontweight='bold')

ax.set_xlabel('Good Volume Declined (%)', fontsize=12)
ax.set_ylabel('Bad Rate Reduction (percentage points)', fontsize=12)
ax.set_title('Policy Rule Efficiency: Bad Rate Reduction vs. Good Volume Cost', fontsize=14)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.legend(['Individual Rules', 'Combined Rulesets'], loc='lower right')
plt.tight_layout()
plt.savefig('figures/efficient_frontier.png', dpi=150, bbox_inches='tight')
```

**Commentary:** "Points in the upper-left are the most efficient — achieving the most bad rate reduction per unit of good volume declined. Ruleset D sits on the efficient frontier, making it our recommended policy package."

### Cell 26 — Waterfall Chart (Optional, for slides)
```python
# Show how bad rate drops as each rule is layered on
# Start from baseline, add rules one by one in order of efficiency
```

---

## Phase 6: Bonus Extensions (if time allows)

### Cell 27 — Vintage Analysis
```python
# For each origination year, track the bad rate
vintage_summary = df.groupby('issue_year').agg(
    n_loans=('default','count'),
    bad_rate=('default','mean'),
    avg_fico=('fico_mid','mean'),
    avg_dti=('dti','mean'),
    total_funded=('loan_amnt','sum'),
).reset_index()

# Plot: bad rate by vintage with credit quality overlaid
fig, ax1 = plt.subplots(figsize=(12,5))
ax2 = ax1.twinx()
ax1.bar(vintage_summary['issue_year'], vintage_summary['bad_rate'],
        color=COLOR_BAD, alpha=0.6, label='Bad Rate')
ax2.plot(vintage_summary['issue_year'], vintage_summary['avg_fico'],
         color=COLOR_GOOD, marker='o', label='Avg FICO')
ax1.set_ylabel('Bad Rate')
ax2.set_ylabel('Avg FICO Score')
ax1.set_title('Vintage Performance: Bad Rate vs. Average Credit Quality')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
```

**Commentary:** "Vintages from [crisis years] show elevated bad rates even after controlling for credit quality, confirming that macroeconomic conditions introduce risk beyond what borrower attributes capture. This argues for ongoing vintage monitoring alongside static policy rules."

### Cell 28 — Rule Stability Across Vintages
```python
# Test: does the recommended ruleset perform consistently across time?
for year in sorted(df['issue_year'].unique()):
    subset = df[df['issue_year'] == year]
    if len(subset) < 1000:
        continue
    mask = RULESETS['D: FICO<660 + Grade E/F/G + Revol>80%'](subset).fillna(False)
    result = evaluate_swap_set(subset, mask, f'Ruleset D — {year}')
    # Collect and plot bad rate reduction by year
```

**Commentary:** "Ruleset D reduces the bad rate in every vintage tested, with the largest impact during stress periods. This suggests the rules are robust rather than overfitting to a single time period."

---

## Phase 7: Notebook Summary

### Cell 29 — Final Recommendation
```python
# Print the recommended ruleset with all key metrics
print("=" * 60)
print("RECOMMENDED POLICY PACKAGE: Ruleset D")
print("=" * 60)
print("Rules:")
print("  1. Decline if FICO score < 660")
print("  2. Decline if Loan Grade is E, F, or G")
print("  3. Decline if Revolving Utilization > 80%")
print()
# Print the evaluate_swap_set output for Ruleset D
```

---

## Phase 8: Slide Deck (5-8 slides)

### Slide 1 — Title
- "Consumer Credit Policy Analysis"
- "Lending Club 2007–2020 | [Your Name] | [Date]"
- Subtitle: "Identifying and quantifying high-risk borrower segments"

### Slide 2 — Problem Framing & Data Scope
- Headline: "1 in 5 loans charged off — $X billion in losses"
- Key facts: total loans analyzed, time range, baseline bad rate
- Methodology note: completed loans only, origination-time features only
- One simple bar chart: Fully Paid vs. Charged Off counts

### Slide 3 — Key Risk Drivers (2×2 panel)
- Four small charts: bad rate by FICO, DTI, Revolving Utilization, Loan Grade
- Each with a one-line callout: "FICO < 660 defaults at 2× the portfolio average"
- Consistent formatting: gray bars for volume, red line for bad rate, dashed line for portfolio average

### Slide 4 — High-Risk Segments (Interaction Heatmap)
- FICO × DTI heatmap with bad rates annotated
- Highlight the worst cell(s) with a box/callout
- "Borrowers with FICO < 660 AND DTI > 35% default at X% — nearly Yx the average"
- Include volume to show it's a meaningful segment, not a small-sample anomaly

### Slide 5 — Proposed Policy Rules
- Clean table listing 3 rules with thresholds and business rationale:
  | Rule | Threshold | Rationale |
  |------|-----------|-----------|
  | FICO floor | < 660 | Subprime borrowers default at 2× average |
  | Grade exclusion | E, F, G | Highest-risk tiers with >30% bad rate |
  | Utilization cap | > 80% | Signals credit stress and over-leverage |

### Slide 6 — Trade-Off Analysis (The Money Slide)
- Rule comparison table (subset: baseline + 3-4 best rulesets)
- Efficient frontier scatter plot
- Callout box with recommended ruleset metrics:
  - "Bad rate: X% → Y% (Z% relative reduction)"
  - "Volume declined: W%"
  - "Loss avoided: $Xm"

### Slide 7 — Recommendations & Next Steps
- Implement Ruleset D as first-pass policy filter
- Build predictive scorecard (logistic regression / GBM) for precision underwriting
- Add purpose-specific rules (e.g., tighter thresholds for small business)
- Monitor by vintage quarterly to detect credit quality shifts
- Validate rules on out-of-time holdout before production deployment

### Slide 8 (Optional) — Appendix
- Data dictionary summary
- Missing value handling
- Full rule comparison table
- AI tools disclosure (required by brief)

---

## File Structure
```
project/
├── README.md                  # Setup instructions + AI disclosure
├── analysis.ipynb             # Main notebook
├── figures/                   # Saved charts for slides
│   ├── outcome_distribution.png
│   ├── bad_rate_by_fico.png
│   ├── bad_rate_by_dti.png
│   ├── bad_rate_by_revol.png
│   ├── bad_rate_by_grade.png
│   ├── heatmap_fico_dti.png
│   ├── efficient_frontier.png
│   └── vintage_analysis.png
├── slides.pptx                # Presentation deck
└── archive/
    ├── Loan_status_2007-2020Q3.gzip
    └── LCDataDictionary.xlsx
```

---

## Checklist Before Submission
- [ ] Notebook runs end-to-end without errors (Kernel → Restart & Run All)
- [ ] Only origination-time columns used (no leakage)
- [ ] loan_status filtered to exactly {'Fully Paid', 'Charged Off'}
- [ ] Baseline bad rate ≈ 19.5%
- [ ] Every rule's swap set bad rate > portfolio average (sanity check)
- [ ] Markdown commentary explains "why" at each step, not just "what"
- [ ] Figures saved to figures/ for slide deck
- [ ] Slides tell a story a non-analyst could follow
- [ ] README includes AI tools disclosure
- [ ] ZIP file or GitHub repo ready
