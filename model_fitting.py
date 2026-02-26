import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, log_loss
)

from statsmodels.discrete.discrete_model import Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson


def fit_and_predict(model_name, X_train, y_train, X_test, y_test,
                    alpha=1.0, l1_ratio=0.5, task_type="regression"):

    # 線形回帰
    if model_name == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, {
            "RMSE": mean_squared_error(y_test, y_pred)**0.5,
            "R2": r2_score(y_test, y_pred)
        }

    # Ridge
    if model_name == "Ridge":
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, {
            "RMSE": mean_squared_error(y_test, y_pred)**0.5,
            "R2": r2_score(y_test, y_pred)
        }

    # Lasso
    if model_name == "Lasso":
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, {
            "RMSE": mean_squared_error(y_test, y_pred)**0.5,
            "R2": r2_score(y_test, y_pred)
        }

    # ElasticNet
    if model_name == "ElasticNet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, {
            "RMSE": mean_squared_error(y_test, y_pred)**0.5,
            "R2": r2_score(y_test, y_pred)
        }

    # ロジスティック回帰
    if model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        return model, y_pred, {
            "Accuracy": accuracy_score(y_test, y_pred),
            "LogLoss": log_loss(y_test, y_pred_proba)
        }

    # プロビット
    if model_name == "Probit":
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        model = Probit(y_train, X_train_sm).fit(disp=False)
        y_pred_proba = model.predict(X_test_sm)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        return model, y_pred, {
            "Accuracy": accuracy_score(y_test, y_pred),
            "LogLoss": log_loss(y_test, y_pred_proba)
        }

    # ポアソン
    if model_name == "Poisson":
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        model = GLM(y_train, X_train_sm, family=Poisson()).fit()
        y_pred = model.predict(X_test_sm)
        return model, y_pred, {
            "RMSE": mean_squared_error(y_test, y_pred)**0.5
        }

    # 決定木（回帰）
    if model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, {
            "RMSE": mean_squared_error(y_test, y_pred)**0.5,
            "R2": r2_score(y_test, y_pred)
        }

    # 決定木（分類）
    if model_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, y_pred, {
            "Accuracy": accuracy_score(y_test, y_pred)
        }

    raise ValueError(f"Unknown model: {model_name}")
