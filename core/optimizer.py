import numpy as np
import pandas as pd
from scipy.optimize import minimize

def generate_obstacles(n, width=500, height=400, radius=30):
    """Return list of random obstacle centers & radii."""
    centers = np.random.randint(radius, min(width, height)-radius, size=(n, 2))
    radii = np.random.randint(radius-10, radius+10, size=n)
    return list(zip(centers, radii))

def cost_function(path_points, start, end):
    """Euclidean path length."""
    pts = np.vstack([start, path_points.reshape(-1, 2), end])
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

def constraint_penalty(path_points, obstacles):
    """Penalty for entering obstacles."""
    penalty = 0
    pts = path_points.reshape(-1, 2)
    for (center, r) in obstacles:
        dist = np.linalg.norm(pts - center, axis=1)
        violation = np.maximum(0, r - dist)
        penalty += np.sum(violation ** 2)
    return penalty

def lagrangian_optimizer(start, end, obstacles, n_points=10, lam=5.0):
    """
    Run constrained optimization using Lagrange-style penalty.
    Returns optimized intermediate points, total cost, and λ values.
    """
    x0 = np.linspace(start[0], end[0], n_points+2)[1:-1]
    y0 = np.linspace(start[1], end[1], n_points+2)[1:-1]
    init = np.vstack([x0, y0]).T.flatten()

    def objective(p):
        base = cost_function(p, start, end)
        penalty = constraint_penalty(p, obstacles)
        return base + lam * penalty

    res = minimize(objective, init, method='L-BFGS-B')
    opt_points = res.x.reshape(-1, 2)
    final_cost = cost_function(res.x, start, end)
    avg_lambda = lam * np.mean(constraint_penalty(res.x, obstacles))

    # log data
    log_df = pd.DataFrame([{
        "start_x": start[0], "start_y": start[1],
        "end_x": end[0], "end_y": end[1],
        "num_obstacles": len(obstacles),
        "cost": round(final_cost, 2),
        "lambda_avg": round(avg_lambda, 4)
    }])
    log_df.to_csv("data/optimization_logs.csv",
                  mode="a", header=False, index=False)

    return opt_points, final_cost, avg_lambda
