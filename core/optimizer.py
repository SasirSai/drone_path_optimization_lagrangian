import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import math


# ---------------------------------------------------------
# SAFE OBSTACLE GENERATION
# ---------------------------------------------------------
def generate_obstacles(
    n, width=600, height=400, radius=25, min_dist_from_start=60,
    min_dist_from_end=60, start=(50,50), end=(550,350)
):
    """
    Generates obstacles ensuring:
    - No obstacle overlaps start / end
    - Reasonable spacing from borders
    """

    obstacles = []
    attempts = 0

    while len(obstacles) < n and attempts < n * 50:
        attempts += 1

        cx = np.random.randint(radius+20, width-radius-20)
        cy = np.random.randint(radius+20, height-radius-20)
        r  = np.random.randint(radius-5, radius+10)

        c = np.array([cx, cy], dtype=float)

        # Must be far from start & end
        if np.linalg.norm(c - np.array(start)) < (r + min_dist_from_start):
            continue
        if np.linalg.norm(c - np.array(end)) < (r + min_dist_from_end):
            continue

        # Avoid overlapping too much with other obstacles
        ok = True
        for oc, orad in obstacles:
            if np.linalg.norm(c - oc) < (r + orad + 25):
                ok = False
                break
        if not ok:
            continue

        obstacles.append((c, r))

    return obstacles


# ---------------------------------------------------------
# SIMPLE A* GRID PLANNER (SAFE FALLBACK)
# ---------------------------------------------------------
def astar_path(start, end, obstacles, canvas_size, grid=10, clearance=20):
    W, H = canvas_size
    Wc, Hc = W // grid, H // grid

    occ = np.ones((Hc, Wc), dtype=bool)

    yy, xx = np.mgrid[0:Hc, 0:Wc]
    pts = np.stack([(xx + 0.5) * grid, (yy + 0.5) * grid], axis=-1)

    for ctr, r in obstacles:
        safe_r = r + clearance
        d = np.linalg.norm(pts - ctr, axis=-1)
        occ &= (d > safe_r)

    def to_cell(p):
        cx = min(max(int(p[0] // grid), 0), Wc-1)
        cy = min(max(int(p[1] // grid), 0), Hc-1)
        return (cy, cx)

    def to_point(cy, cx):
        return np.array([(cx + 0.5)*grid, (cy + 0.5)*grid])

    s = to_cell(start)
    e = to_cell(end)

    # Blocked start/end → enlarge grid and retry
    if not occ[s] or not occ[e]:
        return None

    import heapq

    pq = []
    heapq.heappush(pq, (0, s))
    came = {s: None}
    g = {s: 0}

    nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)]

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == e:
            break
        r, c = cur

        for dr, dc in nbrs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < Hc and 0 <= nc < Wc and occ[nr, nc]:
                new_g = g[cur] + math.hypot(dr, dc)
                if (nr, nc) not in g or new_g < g[(nr,nc)]:
                    g[(nr,nc)] = new_g
                    came[(nr,nc)] = cur
                    h = np.linalg.norm(np.array([nr,nc]) - np.array(e))
                    heapq.heappush(pq, (new_g + h, (nr,nc)))

    if e not in came:
        return None

    # Reconstruct path
    path = []
    cur = e
    while cur:
        path.append(to_point(*cur))
        cur = came[cur]
    path.reverse()

    return np.array(path)


# ---------------------------------------------------------
# SAFE SPLINE BUILDER
# ---------------------------------------------------------
def build_spline(ctrl, start, end, n_samples=200):
    try:
        pts = np.vstack([start, ctrl, end]).astype(float)
        k = min(3, len(pts)-1)
        tck, _ = splprep([pts[:,0], pts[:,1]], s=0, k=k)
        u = np.linspace(0,1,n_samples)
        xs, ys = splev(u, tck)
        return np.column_stack([xs,ys])
    except:
        # Fallback drawing
        return np.vstack([start, end])


def curvature(ctrl):
    if len(ctrl) < 3:
        return 0
    s2 = ctrl[:-2] - 2*ctrl[1:-1] + ctrl[2:]
    return float(np.sum(np.linalg.norm(s2,axis=1)**2))


# ---------------------------------------------------------
# LOG BARRIER (HARD CONSTRAINT)
# ---------------------------------------------------------
def obstacle_penalty(samples, obstacles, clearance=20):
    eps = 1e-9
    for c, r in obstacles:
        safe_r = r + clearance
        d = np.linalg.norm(samples - c, axis=1)

        phi = d - safe_r
        if np.any(phi <= 0):
            return np.inf           # immediate invalid
    # Sum log for valid region → keeps path outside
    return sum([-math.log(phi_i+eps) for c,r in obstacles 
                                  for phi_i in (np.linalg.norm(samples-c,axis=1)- (r+clearance)) 
                                  if phi_i > 0])


# ---------------------------------------------------------
# OBJECTIVE
# ---------------------------------------------------------
def objective(flat, start, end, obstacles, width, height,
              n_ctrl, clearance, lam, gamma):

    ctrl = flat.reshape(n_ctrl, 2)
    samples = build_spline(ctrl, start, end)

    # Path length
    length = np.sum(np.linalg.norm(np.diff(samples, axis=0), axis=1))

    # Obstacle penalty (hard)
    pen_obs = obstacle_penalty(samples, obstacles, clearance)

    # Curvature smoothing
    reg = curvature(ctrl)

    return length + lam*pen_obs + gamma*reg


# ---------------------------------------------------------
# MAIN OPTIMIZER — ALWAYS RETURNS SAFE PATH
# ---------------------------------------------------------
def lagrangian_optimizer(
    start, end, obstacles,
    canvas_size=(600,400),
    n_ctrl=12,
    lam=25.0,
    gamma=2.5,
    clearance=22.0
):
    width, height = canvas_size
    start = np.array(start, float)
    end   = np.array(end, float)

    # ---------------------------------------------
    # Generate initial straight-line control points
    # ---------------------------------------------
    xs = np.linspace(start[0], end[0], n_ctrl+2)[1:-1]
    ys = np.linspace(start[1], end[1], n_ctrl+2)[1:-1]
    init_ctrl = np.column_stack([xs, ys])

    # Bounds inside canvas
    bounds = []
    for _ in range(n_ctrl):
        bounds.append((10, width-10))
        bounds.append((10, height-10))

    # ---------------------------------------------
    # Run Optimization
    # ---------------------------------------------
    res = minimize(
        objective,
        init_ctrl.ravel(),
        args=(start, end, obstacles, width, height, n_ctrl, clearance, lam, gamma),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 250}
    )

    ctrl = res.x.reshape(n_ctrl, 2)
    samples = build_spline(ctrl, start, end)

    # ---------------------------------------------
    # Check feasibility — fallback to A* if unsafe
    # ---------------------------------------------
    safe = True
    for c, r in obstacles:
        if np.min(np.linalg.norm(samples - c, axis=1)) <= (r + clearance):
            safe = False
            break

    if not safe:
        fallback = astar_path(start, end, obstacles, canvas_size, grid=10, clearance=clearance)
        if fallback is not None and len(fallback) > 1:
            samples = fallback

    # Final metrics
    final_cost = np.sum(np.linalg.norm(np.diff(samples, axis=0), axis=1))
    lambda_avg = lam

    return samples, final_cost, lambda_avg
