import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score


def compute_aic_bic(y_true, y_pred, k):
    n = len(y_true)
    residual = y_true - y_pred
    rss = np.sum(residual ** 2)

    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + np.log(n) * k

    return aic, bic


def cross_validate(model_name, X, y, fit_func, k=5, task_type="regression",
                   alpha=1.0, l1_ratio=0.5):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model, y_pred, metrics = fit_func(
            model_name,
            X_train, y_train,
            X_test, y_test,
            alpha=alpha,
            l1_ratio=l1_ratio,
            task_type=task_type
        )

        if task_type == "regression":
            scores.append(-metrics["RMSE"])
        elif task_type == "binary":
            scores.append(metrics["Accuracy"])
        elif task_type == "count":
            scores.append(-metrics["RMSE"])

    return np.mean(scores)


def select_best_model(model_names, X, y, fit_func, k=5, task_type="regression",
                      alpha=1.0, l1_ratio=0.5):

    cv_scores = {}

    for name in model_names:
        score = cross_validate(
            name, X, y,
            fit_func=fit_func,
            k=k,
            task_type=task_type,
            alpha=alpha,
            l1_ratio=l1_ratio
        )
        cv_scores[name] = score

    best_model = max(cv_scores, key=cv_scores.get)

    return best_model, cv_scores
