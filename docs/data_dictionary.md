# Data Dictionary
Source: Kaggle Credit Risk Dataset (CC0). 32,581 loans originally, 30,941 after cleaning.

## Borrower columns
- person_age: age in years
- person_income: annual income in dollars
- person_home_ownership: RENT, OWN, MORTGAGE, or OTHER
- person_emp_length: years employed
- cb_person_default_on_file: Y if a past default is on the credit file, else N
- cb_person_cred_hist_length: credit history length in years

## Loan columns
- loan_intent: purpose (EDUCATION, MEDICAL, VENTURE, PERSONAL, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
- loan_grade: lender risk grade from A (safest) to G (riskiest)
- loan_amnt: loan amount in dollars
- loan_int_rate: interest rate in percent
- loan_percent_income: loan amount divided by annual income
- loan_status: the target. 1 means the borrower defaulted, 0 means the loan was repaid