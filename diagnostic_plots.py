import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def regression_diagnostics(model):
    fitted = model.fittedvalues
    residuals = model.resid
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]

    figs = []

    # 1. 残差 vs Fitted
    fig1, ax1 = plt.subplots()
    ax1.scatter(fitted, residuals, alpha=0.6)
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    figs.append(fig1)

    # 2. Q-Q プロット
    fig2 = sm.qqplot(residuals, line="45", fit=True)
    fig2.suptitle("Normal Q-Q Plot")
    figs.append(fig2)

    # 3. Scale-Location
    fig3, ax3 = plt.subplots()
    ax3.scatter(fitted, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
    ax3.set_xlabel("Fitted values")
    ax3.set_ylabel("√|Standardized Residuals|")
    ax3.set_title("Scale-Location Plot")
    figs.append(fig3)

    # 4. レバレッジ vs 標準化残差
    fig4, ax4 = plt.subplots()
    ax4.scatter(leverage, standardized_residuals, alpha=0.6)
    ax4.set_xlabel("Leverage")
    ax4.set_ylabel("Standardized Residuals")
    ax4.set_title("Residuals vs Leverage (with Cook's distance)")

    for level in [0.5, 1]:
        ax4.axhline(level, color="red", linestyle="--", alpha=0.5)
        ax4.axhline(-level, color="red", linestyle="--", alpha=0.5)

    figs.append(fig4)

    return figs
