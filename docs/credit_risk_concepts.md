# Credit Risk Concepts

## Default
A borrower fails to repay a loan as agreed. In this dataset that is loan_status = 1.

## PD, LGD, EAD, and expected loss
PD (probability of default) is the chance a borrower defaults over a period. LGD (loss given default) is the share of the exposure lost if default happens. EAD (exposure at default) is the amount owed at the time of default. Expected loss = PD x LGD x EAD. This project's model estimates PD only.

## Precision and recall in lending
Recall is the share of real defaults the model catches. Precision is the share of flagged loans that really default. A lender worried about missed defaults wants high recall, at the cost of rejecting more good borrowers.

## Odds ratio
The exponential of a logistic regression coefficient. It shows how much the odds of default are multiplied for a one unit increase, or for a category compared with the baseline. In the statsmodels model, grade A is the baseline.

## Loan grade
The lender's risk rating. In this dataset, default rates rise at every step from grade A to grade G.