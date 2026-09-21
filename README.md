# Credit Risk Default Analysis

A logistic regression analysis identifying which borrower and loan 
characteristics predict loan default, using a dataset of ~31,000 
loan records.

## Tools Used
- SQL (SQLite) — data cleaning, outlier removal, default rate aggregation by segment
- Python (statsmodels) — logistic regression modeling and interpretation
- Python (scikit-learn)

## Status
**Phase 1:** SQL cleaning and aggregation, statsmodels logistic 
regression with odds ratio interpretation.
**Phase 2:** scikit-learn predictive model with expanded 
features, train/test split, and accuracy evaluation.

## Dataset
Credit Risk Dataset, sourced from [[Kaggle]](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). 
Not redistributed here due to licensing; download directly from the 
source to reproduce this analysis.

## Business Problem
Lenders need to understand which borrower and loan characteristics are 
most predictive of default, so risk-based pricing and approval decisions 
can be made with statistical grounding rather than intuition alone.

## Method
Using SQL, I cleaned the dataset by removing data entry errors (unrealistic 
ages above 100, employment lengths exceeding plausible bounds relative to 
age), then calculated default rates by loan grade, loan intent, and home 
ownership status. In Python, I fit a logistic regression model (statsmodels) 
predicting loan default from loan grade, borrower income, and loan amount, 
then converted coefficients to odds ratios for interpretability.

## Key Findings
- Default risk increases sharply and consistently with loan grade: Grade B 
  borrowers are about 1.6x more likely to default than Grade A, rising to 
  roughly 14x for Grade D, 20x for Grade E, and 27x for Grade F, all 
  statistically significant (p < 0.001).
- Grade G showed a directionally similar but highly unstable estimate, 
  driven by a small sample size (n=60), the exact magnitude shouldn't be 
  treated as precise, though the direction (highest risk) is consistent 
  with the overall trend.
- Higher income was associated with modestly lower default odds; larger 
  loan amounts were associated with modestly higher default odds, both 
  holding grade constant.
- The loan grading system appears well-calibrated: default rates from raw 
  SQL aggregation increased monotonically from Grade A through G, and this 
  held up even after controlling for income and loan amount in the 
  regression.

## Recommendation
The strong, monotonic relationship between loan grade and default risk 
suggests the current grading system is a reliable signal for pricing and 
underwriting decisions. I'd recommend maintaining grade-based risk pricing 
as the primary lever, while flagging Grade G as an area needing more data 
before its risk premium is finalized, given the small sample size 
underlying that estimate.

## Files
- `credit_risk_analysis.ipynb` — full notebook (SQL cleaning/aggregation, 
  statsmodels logistic regression)

## Next Steps
- Expand model with additional features (home ownership, loan intent, 
  employment length, credit history length) using scikit-learn
- Train/test split and accuracy evaluation
- Confusion matrix and precision/recall analysis
