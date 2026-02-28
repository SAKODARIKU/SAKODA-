# 最適モデル選択サポート付きデータ分析アプリ

このアプリは、**回帰分析・分類分析・変数選択・変数変換・モデル比較・モデル保存/読み込み** を  
GUI（Streamlit）で簡単に行えるデータ分析支援ツールです。

CSV をアップロードするだけで、誰でも高度なモデル選択ができます。

ベータ版のアプリのurlはこちらですです。
https://zdyz3atpzenwitvjtwj7fy.streamlit.app/
---

## 🚀 機能一覧

### 🔹 1. データ読み込み
- CSV ファイルをアップロードしてデータを表示
- 欠損値の自動処理（X と y の整合性を保つ）

### 🔹 2. 変数選択（Variable Selection）
以下の手法から選択可能：

- **なし（手動選択）**
- **Best Subset Selection（全探索）**
- **Stepwise（AIC/BIC）**
- **LassoCV による自動選択**

選択された変数は後続の変換・モデル学習に反映されます。

### 🔹 3. 変数変換（Transformations）
選択した説明変数に対して以下の変換を適用可能：

- log
- square（2 乗）
- sqrt（平方根）
- standardize（標準化）
- 任意の交互作用（x1 × x2）

### 🔹 4. モデル選択（Model Selection）
以下のモデルを利用可能：

#### 回帰
- Linear Regression  
- Ridge  
- Lasso  
- ElasticNet  
- DecisionTreeRegressor  
- Poisson 回帰  

#### 分類
- Logistic Regression  
- Probit  
- DecisionTreeClassifier  

### 🔹 5. モデル評価
- ホールドアウト評価（MAE, MSE, RMSE, R² など）
- 交差検証（CV）による自動モデル選択
- AIC / BIC の計算
- 正則化後の「生き残った変数のみ」でモデル式を表示

### 🔹 6. モデル保存・読み込み
- 学習したモデルを `.pkl` 形式で保存
- 保存したモデルを読み込み、新しいデータで予測可能

保存先は `saved_models/`（自動生成）です。

### 🔹 7. 回帰診断（Regression Diagnostics）
線形モデルの場合：

- 残差 vs フィット値
- QQ プロット
- 残差ヒストグラム
- Cook’s distance など

---

## 🛠️ インストール

このリポジトリを clone した後、必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

その後、Python上のターミナルに次のように打ち込み、実行すれば起動します。

```bash
streamlit run regression_app.py
