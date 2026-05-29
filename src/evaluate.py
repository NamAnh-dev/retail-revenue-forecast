import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }


def time_series_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = {"MAE": [], "RMSE": [], "MAPE": []}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = compute_metrics(y_test.values, preds)
        for k, v in metrics.items():
            fold_results[k].append(v)

        if verbose:
            print(
                f"  Fold {fold}/{n_splits} — "
                f"MAE: {metrics['MAE']:>8,.0f} | "
                f"RMSE: {metrics['RMSE']:>8,.0f} | "
                f"MAPE: {metrics['MAPE']:>5.1f}%"
            )

    summary = {}
    for k in ["MAE", "RMSE", "MAPE"]:
        summary[k]             = fold_results[k]
        summary[f"mean_{k}"]   = np.mean(fold_results[k])
        summary[f"std_{k}"]    = np.std(fold_results[k])

    return summary


def baseline_last_week(daily: pd.DataFrame) -> dict:
    valid = daily.dropna(subset=["lag_7"])
    metrics = compute_metrics(valid["revenue"].values, valid["lag_7"].values)
    print(f"  Baseline (lag_7) — MAE: {metrics['MAE']:,.0f} | MAPE: {metrics['MAPE']:.1f}%")
    return metrics


# ── Visualization ──────────────────────────────────────────────────────────
def plot_model_comparison(results: dict[str, dict], save_path: str | None = None):
    models = list(results.keys())
    mae    = [results[m]["mean_MAE"]  for m in models]
    rmse   = [results[m]["mean_RMSE"] for m in models]
    mape_  = [results[m]["mean_MAPE"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

    for ax, vals, title, fmt in zip(
        axes,
        [mae, rmse, mape_],
        ["MAE (BRL)", "RMSE (BRL)", "MAPE (%)"],
        ["{:,.0f}", "{:,.0f}", "{:.1f}%"],
    ):
        bars = ax.bar(models, vals, color=colors[:len(models)], width=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                fmt.format(val),
                ha="center", va="bottom", fontsize=10,
            )

    plt.suptitle("Model Comparison — TimeSeriesCV", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_predictions(
    daily: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_col: str = "revenue",
    last_n_days: int = 120,
    save_path: str | None = None,
):

    daily = daily.dropna(subset=feature_cols + [target_col]).copy()
    split = len(daily) - last_n_days

    X_train = daily.iloc[:split][feature_cols]
    y_train = daily.iloc[:split][target_col]
    X_test  = daily.iloc[split:][feature_cols]
    y_test  = daily.iloc[split:][target_col]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = compute_metrics(y_test.values, preds)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"].iloc[split:], y_test.values, label="Actual",    color="#378ADD", alpha=0.8)
    ax.plot(daily["date"].iloc[split:], preds,          label="Predicted", color="#D85A30", alpha=0.9)
    ax.fill_between(
        daily["date"].iloc[split:], y_test.values, preds,
        alpha=0.12, color="#D85A30"
    )
    ax.set_title(
        f"Actual vs Predicted — Last {last_n_days} days\n"
        f"MAE: {metrics['MAE']:,.0f} BRL | MAPE: {metrics['MAPE']:.1f}%",
        fontsize=13,
    )
    ax.set_ylabel("Revenue (BRL)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return metrics


def plot_feature_importance(
    model,
    feature_cols: list[str],
    top_n: int = 15,
    save_path: str | None = None,
):
    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance").tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(importance["feature"], importance["importance"], color="#1D9E75", height=0.6)
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)

    for bar in bars:
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
