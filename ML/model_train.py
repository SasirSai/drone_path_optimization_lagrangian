import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from ml.data_handler import load_and_prepare_data


# ============================================================
#              EXTRA ML EVALUATION GRAPHS
# ============================================================
def plot_prediction_comparison(y_true, y_pred, save_path, r2_value):
    """
    Generates a detailed Predicted vs Actual comparison scatter plot.
    Includes diagonal reference line and R² annotation.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    # --- Scatter points ---
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="black")

    # --- Diagonal y=x line ---
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v],
             "r--", linewidth=2, label="Ideal Fit (y = x)")

    # --- Labels & title ---
    plt.title(f"Predicted vs Actual Path Costs\nR² = {r2_value:.3f}",
              fontsize=14)
    plt.xlabel("Actual Cost", fontsize=12)
    plt.ylabel("Predicted Cost", fontsize=12)
    plt.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✅ Saved: {save_path}")

def plot_pred_vs_actual(y_true, y_pred, save_path, r2):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    # Diagonal reference line
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Ideal Fit")

    plt.title(f"Predicted vs Actual Costs (R² = {r2:.3f})", fontsize=14)
    plt.xlabel("Actual Cost")
    plt.ylabel("Predicted Cost")
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path, dpi=150)
    plt.close()
    print("✅ Saved:", save_path)


def plot_residuals(y_true, y_pred, save_path):
    residuals = y_true - y_pred

    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')

    plt.title("Residual Plot (Error vs Predicted)", fontsize=14)
    plt.xlabel("Predicted Cost")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True)

    plt.savefig(save_path, dpi=150)
    plt.close()
    print("✅ Saved:", save_path)


def plot_error_histogram(y_true, y_pred, save_path):
    errors = y_true - y_pred

    plt.figure(figsize=(8,6))
    plt.hist(errors, bins=30, edgecolor="black", alpha=0.7)

    plt.title("Prediction Error Distribution", fontsize=14)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig(save_path, dpi=150)
    plt.close()
    print("✅ Saved:", save_path)



# ============================================================
#                   MAIN TRAINING FUNCTION
# ============================================================

def train_and_save_model():

    # Load dataset (NO CHANGES)
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Keep Linear Regression (NO CHANGES)
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    # Save model
    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, "ml/model.pkl")

    print("✅ Model Trained")
    print(f"MAE: {mae:.3f}")
    print(f"R² : {r2:.3f}")
    print("✅ Saved → ml/model.pkl")

    os.makedirs("results", exist_ok=True)

    # ✅ A) Enhanced Predicted vs Actual
    plot_pred_vs_actual(y_test, preds,
                        "results/predicted_vs_actual.png",
                        r2)

    # ✅ B) Residual Plot
    plot_residuals(y_test, preds,
                   "results/residual_plot.png")

    # ✅ C) Error Histogram
    plot_error_histogram(y_test, preds,
                         "results/error_histogram.png")

    print("\n✅ ALL evaluation plots saved in /results folder")


# Run directly
if __name__ == "__main__":
    train_and_save_model()
