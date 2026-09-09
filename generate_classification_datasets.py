# -*- coding: utf-8 -*-
"""Generate the two classification datasets used by Data with Roots - Activity 2.

1) data/logistic_churn.csv      -> Binary Logistic Regression (1 independent variable)
     X: tenure_months | Y: churn (0 = Stay, 1 = Churn)
2) data/sgd_credit_risk.csv     -> SGD Classifier / Assigned Model (4 independent variables)
     X: income_usd, loan_amount_usd, credit_score, months_employed
     Y: high_risk (0 = Low Risk, 1 = High Risk)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- logistic_churn
n = 600
tenure = rng.uniform(1, 60, n)
p_churn = sigmoid(3.0 - 0.11 * tenure)
churn = np.where(rng.random(n) < p_churn, 1, 0)
logistic_df = pd.DataFrame({
    "tenure_months": np.round(tenure, 1),
    "churn": churn,
})
logistic_df.to_csv("data/logistic_churn.csv", index=False)

# ---------------------------------------------------------------- sgd_credit_risk
income = rng.uniform(500, 6000, n)
loan = rng.uniform(1000, 40000, n)
credit_score = rng.uniform(300, 850, n)
months_employed = rng.uniform(0, 300, n)

linear = (0.9 * (income / 6000)
          + 1.1 * (loan / 40000)
          + 2.2 * (1 - credit_score / 850)
          + 0.8 * (1 - np.minimum(months_employed / 300.0, 1.0))
          - 2.2)
p_high = sigmoid(3.0 * linear)
high_risk = np.where(rng.random(n) < p_high, 1, 0)

credit_df = pd.DataFrame({
    "income_usd": np.round(income, 2),
    "loan_amount_usd": np.round(loan, 2),
    "credit_score": credit_score.astype(int),
    "months_employed": np.round(months_employed, 1),
    "high_risk": high_risk,
})
credit_df.to_csv("data/sgd_credit_risk.csv", index=False)

# ------------------------------------------------- sanity metrics of both datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=== logistic_churn.csv ===")
print(logistic_df.head(3))
print("class balance:", logistic_df["churn"].value_counts().to_dict())

X = logistic_df[["tenure_months"]].values
y = logistic_df["churn"].values
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=2000)
lr.fit(Xt, yt)
print("ACC %.3f PRE %.3f REC %.3f F1 %.3f" % (
    accuracy_score(yv, lr.predict(Xv)),
    precision_score(yv, lr.predict(Xv)),
    recall_score(yv, lr.predict(Xv)),
    f1_score(yv, lr.predict(Xv))))

print("=== sgd_credit_risk.csv ===")
print(credit_df.head(3))
print("class balance:", credit_df["high_risk"].value_counts().to_dict())

Xc = credit_df[["income_usd", "loan_amount_usd", "credit_score", "months_employed"]].values
yc = credit_df["high_risk"].values
Xct, Xcv, yct, ycv = train_test_split(Xc, yc, test_size=0.2, random_state=42)
sgd = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", max_iter=5000, random_state=42))
sgd.fit(Xct, yct)
print("ACC %.3f PRE %.3f REC %.3f F1 %.3f" % (
    accuracy_score(ycv, sgd.predict(Xcv)),
    precision_score(ycv, sgd.predict(Xcv)),
    recall_score(ycv, sgd.predict(Xcv)),
    f1_score(ycv, sgd.predict(Xcv))))