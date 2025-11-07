import os
import joblib
import numpy as np

def predict_path_cost(start, end, num_obstacles, lambda_avg=3.0):
    """
    Predict optimal path cost using the trained regression model.
    start, end: tuples (x, y)
    num_obstacles: int
    lambda_avg: float (use last optimization's λ_avg if available; else default 3.0)
    """
    model_path = "ml/model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train it first (ml/model_train.py).")

    model = joblib.load(model_path)

    X_new = np.array([[float(start[0]), float(start[1]),
                       float(end[0]),   float(end[1]),
                       float(num_obstacles), float(lambda_avg)]])
    pred = float(model.predict(X_new)[0])
    return pred
