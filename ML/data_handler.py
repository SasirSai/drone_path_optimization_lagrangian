import os
import pandas as pd
from sklearn.model_selection import train_test_split

COLS = ["start_x","start_y","end_x","end_y","num_obstacles","cost","lambda_avg"]

def _ensure_min_dataset(df: pd.DataFrame, min_rows: int = 24) -> pd.DataFrame:
    """
    If logs are tiny, augment by jittering start/end & lambda to create a small
    but valid training set. This keeps the pipeline usable during early testing.
    """
    if len(df) >= min_rows:
        return df.reset_index(drop=True)

    if df.empty:
        # synthesize a tiny dataset if nothing exists yet
        rows = []
        import random
        for _ in range(min_rows):
            sx, sy = random.randint(30, 120), random.randint(30, 120)
            ex, ey = random.randint(420, 580), random.randint(280, 380)
            k = random.randint(1, 5)
            lam = round(random.uniform(1.5, 5.0), 3)
            # naive distance baseline as proxy for cost
            dist = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
            cost = dist * (1.0 + 0.02*k) * (1.0 + 0.02*(lam-3.0))
            rows.append([sx, sy, ex, ey, k, round(cost, 2), lam])
        synth = pd.DataFrame(rows, columns=COLS)
        return synth

    # jitter the small log
    import numpy as np
    needed = max(0, min_rows - len(df))
    base = df.sample(n=min(len(df), max(1, len(df))), replace=True).copy()
    reps = []
    for _ in range(needed):
        r = base.sample(1).iloc[0].copy()
        # jitter features slightly
        r["start_x"] += int(np.random.randint(-8, 9))
        r["start_y"] += int(np.random.randint(-8, 9))
        r["end_x"]   += int(np.random.randint(-8, 9))
        r["end_y"]   += int(np.random.randint(-8, 9))
        r["lambda_avg"] = float(max(0.5, r["lambda_avg"] + float(np.random.randn()*0.2)))
        # recompute a proxy cost to keep relationships sane
        dist = ((r["end_x"] - r["start_x"])**2 + (r["end_y"] - r["start_y"])**2) ** 0.5
        k = max(1, int(r["num_obstacles"]))
        r["cost"] = round(dist * (1.0 + 0.02*k) * (1.0 + 0.02*(r["lambda_avg"]-3.0)), 2)
        reps.append(r)
    df_aug = pd.concat([df, pd.DataFrame(reps)], ignore_index=True)
    return df_aug.reset_index(drop=True)

def load_and_prepare_data(path: str = "data/training_data.csv"):
    """
    Loads the MAIN dataset (3000+ rows) generated earlier.
    Does NOT augment or reduce data.
    Returns proper train/test split.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Dataset not found at {path}.\n"
            "Make sure your 3000-row training_data.csv is inside /data."
        )

    df = pd.read_csv(path)

    # Select the correct columns
    X = df[["start_x","start_y","end_x","end_y","num_obstacles","lambda_avg"]].astype(float)
    y = df["cost"].astype(float)

    return train_test_split(X, y, test_size=0.25, random_state=42)
