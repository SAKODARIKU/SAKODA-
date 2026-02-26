import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from variable_selection import (
    best_subset_selection,
    stepwise_selection,
    lasso_variable_selection
)

from model_fitting import fit_and_predict
from model_evaluation import (
    compute_aic_bic,
    cross_validate,
    select_best_model
)

from diagnostic_plots import regression_diagnostics
from probit_visualization import probit_effect_plot


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="最適モデル選択アプリ", layout="wide")
st.title("最適モデル選択サポート付きデータ分析アプリ")


# ------------------------------------------------------------
# サイドバー：データ入力
# ------------------------------------------------------------
st.sidebar.header("データ入力")

uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is None:
    st.info("左のサイドバーから CSV をアップロードしてください")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### データプレビュー")
st.dataframe(df)


# ------------------------------------------------------------
# サイドバー：目的変数・説明変数
# ------------------------------------------------------------
st.sidebar.header("変数設定")

target = st.sidebar.selectbox("目的変数（y）", df.columns)
features_all = st.sidebar.multiselect(
    "説明変数（X）", df.columns.drop(target)
)

if len(features_all) == 0:
    st.warning("説明変数を 1 つ以上選んでください")
    st.stop()

y = df[target]
X_raw = df[features_all].select_dtypes(include=[np.number]).dropna()
y = y.loc[X_raw.index]

# X と y を同時に dropna（NaN 行を完全に除去）
data = pd.concat([X_raw, y], axis=1).dropna()
X_raw = data[features_all]
y = data[target]


# ------------------------------------------------------------
# サイドバー：変数選択
# ------------------------------------------------------------
st.sidebar.header("変数選択")

var_sel_method = st.sidebar.selectbox(
    "変数選択方法",
    ["none", "best_subset", "stepwise_AIC", "stepwise_BIC", "lasso"],
    format_func=lambda x: {
        "none": "なし",
        "best_subset": "全探索（best subset）",
        "stepwise_AIC": "ステップワイズ（AIC）",
        "stepwise_BIC": "ステップワイズ（BIC）",
        "lasso": "Lasso による自動選択"
    }[x]
)

selected_features = list(X_raw.columns)
var_sel_info = ""

if var_sel_method == "best_subset":
    max_features = min(10, len(selected_features))
    selected_features, best_aic = best_subset_selection(X_raw, y, max_features)
    var_sel_info = f"best subset により選択: {selected_features}（AIC={best_aic:.3f}）"

elif var_sel_method in ["stepwise_AIC", "stepwise_BIC"]:
    criterion = "AIC" if var_sel_method == "stepwise_AIC" else "BIC"
    selected_features, score = stepwise_selection(X_raw, y, criterion=criterion)
    var_sel_info = f"ステップワイズ（{criterion}）により選択: {selected_features}"

elif var_sel_method == "lasso":
    selected_features, lasso_model = lasso_variable_selection(X_raw, y)
    var_sel_info = f"Lasso により選択: {selected_features}"

else:
    var_sel_info = "変数選択なし（手動選択）"

st.write("### 変数選択結果")
st.write(var_sel_info)

X = X_raw[selected_features]



# ------------------------------------------------------------
# サイドバー：変数変換（全部入り）
# ------------------------------------------------------------
st.sidebar.header("変数変換")

# 変換対象の変数を選ぶ
transform_targets = st.sidebar.multiselect(
    "変換する変数を選択（複数可）",
    selected_features
)

# 変換方法を選ぶ
transform_methods = st.sidebar.multiselect(
    "適用する変換を選択",
    ["log", "square", "sqrt", "standardize"]
)

# 交互作用の追加
interaction_pairs = st.sidebar.multiselect(
    "交互作用（x1 × x2）を追加",
    [(a, b) for a in selected_features for b in selected_features if a < b],
    format_func=lambda x: f"{x[0]} × {x[1]}"
)

# ---- 変換を適用 ----
X_transformed = X_raw[selected_features].copy()

# 個別変換
for col in transform_targets:
    if "log" in transform_methods:
        X_transformed[f"log_{col}"] = np.log(X_transformed[col].replace(0, np.nan))
    if "square" in transform_methods:
        X_transformed[f"{col}_sq"] = X_transformed[col] ** 2
    if "sqrt" in transform_methods:
        X_transformed[f"sqrt_{col}"] = np.sqrt(X_transformed[col])
    if "standardize" in transform_methods:
        X_transformed[f"std_{col}"] = (X_transformed[col] - X_transformed[col].mean()) / X_transformed[col].std()

# 交互作用
for a, b in interaction_pairs:
    X_transformed[f"{a}_x_{b}"] = X_transformed[a] * X_transformed[b]

# 変換後の説明変数一覧を更新
all_transformed_features = list(X_transformed.columns)

# 変換後のデータを X に反映
X = X_transformed

# 変換後に NaN 行を落とす（X と y をそろえて）
data2 = pd.concat([X, y], axis=1).dropna()
X = data2[X.columns]
y = data2[y.name]



# ------------------------------------------------------------
# サイドバー：モデル選択
# ------------------------------------------------------------
st.sidebar.header("モデル選択")

# ---- モデル選択を完全に自由化 ----
task_type = "free"

model_choices = [
    "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "DecisionTreeRegressor",
    "LogisticRegression", "Probit", "DecisionTreeClassifier",
    "Poisson"
]


model_sel_method = st.sidebar.selectbox(
    "モデル選択方法",
    ["manual", "auto_cv"],
    format_func=lambda x: {
        "manual": "手動",
        "auto_cv": "自動（交差検証）"
    }[x]
)


if model_sel_method == "manual":
    selected_model_name = st.sidebar.selectbox("モデルを選択", model_choices)
    model_names_to_run = [selected_model_name]
else:
    st.sidebar.write("以下のモデルから最良を自動選択：")
    st.sidebar.write(model_choices)
    model_names_to_run = model_choices


# ------------------------------------------------------------
# サイドバー：パラメータ
# ------------------------------------------------------------
st.sidebar.header("パラメータ")

alpha = st.sidebar.slider("α（正則化）", 0.01, 10.0, 1.0)
l1_ratio = st.sidebar.slider("ElasticNet の L1 比率", 0.0, 1.0, 0.5)
k_cv = st.sidebar.slider("交差検証 K", 3, 10, 5)


# ------------------------------------------------------------
# モデル学習（ホールドアウト）
# ------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models_fitted = {}
holdout_metrics = {}

# ---- モデル実行の安全チェック ----
def is_valid_for_model(model_name, y):
    vals=y.dropna().unique()
    # Probit / Logistic は 0/1 のみ
    if model_name in ["Probit", "LogisticRegression"]:
        return set(vals) <= {0, 1}

# DecisionTreeClassifier も離散値のみ（連続値は不可） 
    if model_name == "DecisionTreeClassifier": 
        # 連続値が含まれていたら除外 
        return np.all(np.floor(vals) == vals)

    # Poisson は非負整数のみ
    if model_name == "Poisson":
        return (vals >= 0).all() and np.all(np.floor(vals) == vals)

    # その他のモデルは何でもOK
    return True

# 実行可能なモデルだけ残す
model_names_to_run = [m for m in model_names_to_run if is_valid_for_model(m, y)]

if len(model_names_to_run) == 0:
    st.error("このデータでは実行可能なモデルがありません。")
    st.stop()

for name in model_names_to_run:
    model, y_pred, metrics = fit_and_predict(
        name, X_train, y_train, X_test, y_test,
        alpha=alpha, l1_ratio=l1_ratio, task_type=task_type
    )
    models_fitted[name] = model
    holdout_metrics[name] = metrics


# ------------------------------------------------------------
# 自動モデル選択（CV）
# ------------------------------------------------------------
if model_sel_method == "auto_cv":
    best_model_name, cv_scores = select_best_model(
        model_names_to_run, X, y,
        fit_func=fit_and_predict,
        k=k_cv,
        task_type=task_type,
        alpha=alpha,
        l1_ratio=l1_ratio
    )
else:
    best_model_name = model_names_to_run[0]
    cv_scores = {best_model_name: None}

best_model = models_fitted[best_model_name]


# ------------------------------------------------------------
# 結果表示
# ------------------------------------------------------------
st.write("## モデル比較（ホールドアウト & CV）")

rows = []
for name in model_names_to_run:
    row = {"Model": name}
    row.update(holdout_metrics[name])
    row["CV_score"] = cv_scores.get(name, None)
    rows.append(row)

st.dataframe(pd.DataFrame(rows))

st.success(f"選択されたモデル: {best_model_name}")


# ------------------------------------------------------------
# 選択モデルの詳細
# ------------------------------------------------------------
st.write("## 選択モデルの詳細")


st.write("## 保存済みモデルの読み込み")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
saved_dir = os.path.join(BASE_DIR, "saved_models")
os.makedirs(saved_dir, exist_ok=True)
saved_files = [f for f in os.listdir(saved_dir) if f.endswith(".pkl")]

if len(saved_files) == 0:
    st.info("保存されたモデルはまだありません")
else:
    selected_saved_model = st.selectbox("読み込むモデルを選択", saved_files)

    if st.button("モデルを読み込む"):
        with open(os.path.join(saved_dir, selected_saved_model), "rb") as f:
            loaded_model = pickle.load(f)
        st.success(f"{selected_saved_model} を読み込みました！")

# ------------------------------------------------------------
# モデルの保存
# ------------------------------------------------------------
st.write("## モデルの保存")
filename = st.text_input("保存するファイル名（例: model.pkl）", "model.pkl")

if st.button("モデルを保存する"):
    save_path = os.path.join(saved_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)
    st.success(f"モデルを保存しました: {save_path}")


    # 読み込んだモデルで予測
    st.write("### 保存モデルで予測")

    input_values = {}
    for feat in X.columns:
        input_values[feat] = st.number_input(f"{feat} の値を入力（保存モデル用）", value=0.0)

    if st.button("保存モデルで予測する"):
        X_new = np.array([input_values[feat] for feat in X.columns]).reshape(1, -1)
        y_new = loaded_model.predict(X_new)[0]
        st.success(f"予測結果: {y_new:.4f}")


if best_model_name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:

    # モデルが実際に使った説明変数（変換後）
    features = list(X.columns)
    coefs = best_model.coef_

    # 係数が 0 の変数だけ除外した DataFrame（正則化後の生き残り）
    nonzero_mask = np.abs(coefs) > 1e-8
    coef_df = pd.DataFrame({
        "feature": np.array(features)[nonzero_mask],
        "coef": coefs[nonzero_mask]
    })

    # モデル式（係数0の変数は式に含めない）
    terms = [
        f"{coef:.4f} * {feat}"
        for coef, feat in zip(coefs, features)
        if abs(coef) > 1e-8
    ]

    formula = f"y = {best_model.intercept_:.4f}"
    if terms:
        formula += " + " + " + ".join(terms)

    st.write("### モデルの式（正則化・変換後の変数を正しく反映）")
    st.code(formula)

    st.write("### 回帰係数（正則化で残った変数のみ）")
    st.dataframe(coef_df)

    # AIC/BIC
    y_pred = best_model.predict(X_test)
    aic, bic = compute_aic_bic(y_test, y_pred, k=len(coef_df) + 1)
    st.write(f"AIC: {aic:.3f}, BIC: {bic:.3f}")


elif best_model_name == "DecisionTreeRegressor":
    st.write("### 決定木の特徴量重要度")
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    st.dataframe(importance_df)
    st.bar_chart(importance_df.set_index("feature"))

elif best_model_name == "DecisionTreeClassifier":
    st.write("### 決定木の特徴量重要度")
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    st.dataframe(importance_df)
    st.bar_chart(importance_df.set_index("feature"))

elif best_model_name == "Probit":
    st.write("### プロビットモデルの影響度")
    coef_df, fig = probit_effect_plot(best_model, X.columns)
    st.dataframe(coef_df)
    st.pyplot(fig)

elif best_model_name == "Poisson":
    st.write("### ポアソン回帰の係数")
    params = best_model.params
    coef_df = pd.DataFrame({
        "feature": ["const"] + list(X.columns),
        "coef": params.values
    })
    st.dataframe(coef_df)

st.write("### 新しいデータで予測")

input_values = {}
for feat in X.columns:
    input_values[feat] = st.number_input(f"{feat} の値を入力", value=0.0)

if st.button("予測する"):
    X_new = np.array([input_values[feat] for feat in X.columns]).reshape(1, -1)
    y_new = best_model.predict(X_new)[0]
    st.success(f"予測された y の値: {y_new:.4f}")

# ------------------------------------------------------------
# 回帰診断（線形モデルのみ）
# ------------------------------------------------------------
if best_model_name in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
    st.write("## 回帰診断プロット")
    import statsmodels.api as sm
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()

    figs = regression_diagnostics(ols_model)
    for fig in figs:
        st.pyplot(fig)
