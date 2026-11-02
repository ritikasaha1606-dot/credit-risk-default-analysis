# Model Card

## Purpose
Predicts whether a borrower defaults. A portfolio project, not for real lending decisions.

## Data and cleaning
Removed 5 rows with age above 100 and rows where age minus employment length was under 16 (impossible work histories). 30,941 rows remained. Missing interest rates were filled with the median.

## Model
Logistic regression (scikit-learn) on 22 features after one-hot encoding, standardized with StandardScaler, with an 80/20 train/test split. The test set had 6,189 loans. A statsmodels logistic regression on grade, income, and loan amount was used for interpretation.

## Performance
Accuracy is 86.8%. Always predicting no default would score about 78%. For the default class, precision is 0.78 and recall is 0.55: the model caught 748 of 1,357 real defaults and missed 609.

## Limitations
- Recall on defaults is only 55%, and missed defaults are the costly error in lending.
- Grade G has only 60 loans, so its estimates are unreliable.
- The median fill was done before the train/test split, a minor form of data leakage.
- Scikit-learn coefficients are on standardized features, so they are not comparable to the raw-unit statsmodels coefficients.
- Public dataset and a simple linear model, not validated for real use.