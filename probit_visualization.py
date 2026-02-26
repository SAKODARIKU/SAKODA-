import pandas as pd
import matplotlib.pyplot as plt


def probit_effect_plot(model, feature_names):
    params = model.params.values

    coef_df = pd.DataFrame({
        "feature": ["const"] + feature_names,
        "coef": params
    })

    effect_df = coef_df[coef_df["feature"] != "const"].copy()
    effect_df["abs_coef"] = effect_df["coef"].abs()
    effect_df = effect_df.sort_values("abs_coef", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(effect_df["feature"], effect_df["abs_coef"], color="skyblue")
    ax.set_title("Probit Model Feature Influence (|coef|)")
    ax.set_ylabel("Absolute Coefficient")
    ax.set_xticklabels(effect_df["feature"], rotation=45, ha="right")

    return coef_df, fig

