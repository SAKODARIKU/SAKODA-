import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV


# ============================================================
# 1. Best Subset Selection（全探索）
# ============================================================

def best_subset_selection(X, y, max_features=None):
    """
    全探索による最良モデル選択（AIC最小）
    max_features が None の場合は全ての特徴量まで探索
    """
    features = list(X.columns)
    if max_features is None:
        max_features = len(features)

    best_aic = np.inf
    best_subset = features

    for k in range(1, max_features + 1):
        for combo in itertools.combinations(features, k):
            X_sub = sm.add_constant(X[list(combo)])
            model = sm.OLS(y, X_sub).fit()

            if model.aic < best_aic:
                best_aic = model.aic
                best_subset = list(combo)

    return best_subset, best_aic


# ============================================================
# 2. Stepwise Selection（AIC / BIC）
# ============================================================

def stepwise_selection(X, y, criterion="AIC", direction="both"):
    """
    ステップワイズ法（前進・後退・双方向）
    criterion: "AIC" or "BIC"
    direction: "forward", "backward", "both"
    """

    remaining = set(X.columns)
    selected = []
    current_score = np.inf

    def score_model(features):
        if len(features) == 0:
            X_sub = sm.add_constant(pd.DataFrame({"intercept": np.ones(len(y))}))
        else:
            X_sub = sm.add_constant(X[list(features)])
        model = sm.OLS(y, X_sub).fit()
        return model.aic if criterion == "AIC" else model.bic

    # 初期スコア（空モデル）
    current_score = score_model([])

    # -------------------------
    # 前進選択
    # -------------------------
    if direction in ["forward", "both"]:
        improved = True
        while improved and remaining:
            scores = []
            for candidate in remaining:
                score = score_model(selected + [candidate])
                scores.append((score, candidate))

            scores.sort()
            best_new_score, best_candidate = scores[0]

            if best_new_score < current_score:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                current_score = best_new_score
            else:
                improved = False

    # -------------------------
    # 後退選択
    # -------------------------
    if direction in ["backward", "both"]:
        improved = True
        while improved and len(selected) > 0:
            scores = []
            for candidate in list(selected):
                reduced = list(selected)
                reduced.remove(candidate)
                score = score_model(reduced)
                scores.append((score, candidate))

            scores.sort()
            best_new_score, worst_candidate = scores[0]

            if best_new_score < current_score:
                selected.remove(worst_candidate)
                current_score = best_new_score
            else:
                improved = False

    return selected, current_score


# ============================================================
# 3. Lasso による変数選択（LassoCV）
# ============================================================

def lasso_variable_selection(X, y):
    """
    LassoCV による変数選択（α自動選択）
    """
    model = LassoCV(cv=5, random_state=42)
    model.fit(X, y)

    coef = model.coef_
    selected = [col for col, c in zip(X.columns, coef) if abs(c) > 1e-6]

    return selected, model
