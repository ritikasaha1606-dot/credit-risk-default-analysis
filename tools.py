import sqlite3, joblib
import numpy as np
import pandas as pd

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('columns.pkl')
conn = sqlite3.connect('credit_risk.db', check_same_thread=False)

ALLOWED_COLUMNS = ['loan_grade', 'loan_intent',
                   'person_home_ownership', 'cb_person_default_on_file']
VALID = {
    'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
    'loan_intent': ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL',
                    'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
    'loan_grade': list('ABCDEFG'),
    'cb_person_default_on_file': ['Y', 'N'],
}

def default_rate_by(column: str) -> list:
    """Returns the historical default rate and number of loans for each
    category of one column, from the cleaned dataset of about 31,000 loans.

    Use for questions about which group defaults more or less, such as
    'which loan grade is riskiest' or 'do renters default more'.
    Do not use for predicting one specific borrower, or for numeric
    columns like income or age.

    Args:
        column: One of 'loan_grade', 'loan_intent',
            'person_home_ownership', 'cb_person_default_on_file'.

    Returns:
        A list of rows with the category, default_rate (0 to 1),
        and loan_count.
    """
    if column not in ALLOWED_COLUMNS:
        return [{"error": f"Column must be one of {ALLOWED_COLUMNS}"}]
    query = f"""
    SELECT {column},
           AVG(loan_status) AS default_rate,
           COUNT(*) AS loan_count
    FROM credit_risk_clean_2
    GROUP BY {column}
    """
    out = pd.read_sql_query(query, conn)
    out['default_rate'] = out['default_rate'].round(4)
    return out.to_dict('records')


def score_borrower(person_age: int, person_income: float,
                   person_emp_length: float, loan_amnt: float,
                   loan_int_rate: float, cb_person_cred_hist_length: int,
                   person_home_ownership: str, loan_intent: str,
                   loan_grade: str, cb_person_default_on_file: str) -> float:
    """Returns the predicted probability of default (0 to 1) for one
    hypothetical borrower, using the trained logistic regression model.

    Use only when the user gives specific borrower details. If any detail
    is missing, ask the user for it instead of guessing.
    Do not use for group-level questions.

    Args:
        person_age: Age in years.
        person_income: Annual income in dollars.
        person_emp_length: Years employed.
        loan_amnt: Loan amount in dollars.
        loan_int_rate: Interest rate as a percent, e.g. 11.5 for 11.5%.
        cb_person_cred_hist_length: Credit history length in years.
        person_home_ownership: 'RENT', 'OWN', 'MORTGAGE' or 'OTHER'.
        loan_intent: 'EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL',
            'HOMEIMPROVEMENT' or 'DEBTCONSOLIDATION'.
        loan_grade: 'A' (safest) to 'G' (riskiest).
        cb_person_default_on_file: 'Y' if a past default is on file, else 'N'.

    Returns:
        Probability of default between 0 and 1.
    """
    inputs = {
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'cb_person_default_on_file': cb_person_default_on_file,
    }
    for name, value in inputs.items():
        if value not in VALID[name]:
            return f"{name} must be one of {VALID[name]}"
    if person_income <= 0:
        return "person_income must be greater than 0"

    row = dict.fromkeys(feature_cols, 0)
    row['person_age'] = person_age
    row['person_income'] = person_income
    row['person_emp_length'] = person_emp_length
    row['loan_amnt'] = loan_amnt
    row['loan_int_rate'] = loan_int_rate
    row['cb_person_cred_hist_length'] = cb_person_cred_hist_length
    row['loan_percent_income'] = round(loan_amnt / person_income, 2)

    for name, value in inputs.items():
        col = f'{name}_{value}'
        if col in row:
            row[col] = 1

    X_new = pd.DataFrame([row])[feature_cols]
    X_new_scaled = scaler.transform(X_new)
    return round(float(model.predict_proba(X_new_scaled)[0, 1]), 3)


def top_risk_factors(n: int = 10) -> list:
    """Returns the features with the largest effect on predicted default
    risk, ranked by size, from the trained model.

    Use for questions about what drives default or which factors matter
    most. Do not use for a specific borrower's score.
    Note: features were standardized, so odds_ratio_per_1SD means the
    change in odds of default for a 1 standard deviation increase
    (or for a category switched on).

    Args:
        n: How many top features to return. Default 10.

    Returns:
        A list of rows with feature, coefficient and odds_ratio_per_1SD.
    """
    coefs = pd.Series(model.coef_[0], index=feature_cols)
    out = pd.DataFrame({'coefficient': coefs,
                        'odds_ratio_per_1SD': np.exp(coefs)})
    out = out.reindex(out['coefficient'].abs().sort_values(ascending=False).index)
    return out.head(n).round(3).reset_index(names='feature').to_dict('records')