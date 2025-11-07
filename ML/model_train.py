import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from ml.data_handler import load_and_prepare_data

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    os.makedirs("ml", exist_ok=True)
    joblib.dump(model, "ml/model.pkl")

    print(f"Model Trained")
    print(f"MAE: {mae:.3f}")
    print(f"R² : {r2:.3f}")
    print("Saved → ml/model.pkl")

    # Save a simple Predicted vs Actual plot for the report
    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.scatter(y_test, preds)
    plt.xlabel("Actual Cost")
    plt.ylabel("Predicted Cost")
    plt.title("Predicted vs Actual Path Costs")
    plt.grid(True)
    plt.savefig("results/prediction_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    train_and_save_model()
