"""
Runner for Requirement 1 SHAP explainability.
Loads and cleans the hotel booking data, engineers features, trains the
candidate regressors, selects the best by test R2, and produces SHAP artifacts.
"""

from __future__ import annotations

from pathlib import Path

from src.data_loader import get_default_hotel_booking_path, load_hotel_booking_data
from src.data_processing import (
    clean_hotel_booking_data,
    engineer_features,
    prepare_features,
    split_data,
)
from src.shap_analysis import (
    compute_shap_values,
    plot_shap_bar,
    plot_shap_beeswarm,
    select_best_model,
    top_features,
)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_hotel_booking_data(get_default_hotel_booking_path())
    df = clean_hotel_booking_data(df)
    df = engineer_features(df)
    X, y = prepare_features(df, target_column="adr")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Data ready: X_train={X_train.shape}, X_test={X_test.shape}, features={X.shape[1]}")

    best_name, best_model, metrics = select_best_model(X_train, y_train, X_test, y_test)

    print("\n=== Model comparison (test set) ===")
    for name, m in metrics.items():
        marker = "  <-- BEST" if name == best_name else ""
        print(f"  {name:18s} R2={m['R2']:.4f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}{marker}")

    print(f"\nBest model: {best_name}")

    shap_values = compute_shap_values(best_model, X_test)

    bar_path = plot_shap_bar(shap_values, REPORTS_DIR)
    bee_path = plot_shap_beeswarm(shap_values, REPORTS_DIR)
    print(f"\nSaved: {bar_path.name}, {bee_path.name}")

    ranking = top_features(shap_values, n=10)
    print("\n=== Top 10 features by mean |SHAP| ===")
    print(ranking.to_string())


if __name__ == "__main__":
    main()
