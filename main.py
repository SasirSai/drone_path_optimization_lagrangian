import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from core.optimizer import generate_obstacles, lagrangian_optimizer
from gui.visualizer import draw_environment, draw_path

# ---------- GUI SETUP ----------
root = tk.Tk()
root.title("Drone Path Optimization - Lagrangian Method")
root.geometry("800x600")
root.resizable(False, False)

# frames
top_frame = ttk.Frame(root)
top_frame.pack(side=tk.TOP, pady=10)

canvas = tk.Canvas(root, width=600, height=400, bg="white")
canvas.pack(pady=10)

bottom_frame = ttk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, pady=10)

# ---------- INPUT WIDGETS ----------
ttk.Label(top_frame, text="Start (x,y):").grid(row=0, column=0, padx=5)
start_x = ttk.Entry(top_frame, width=5); start_x.insert(0, "50")
start_y = ttk.Entry(top_frame, width=5); start_y.insert(0, "50")
start_x.grid(row=0, column=1); start_y.grid(row=0, column=2)

ttk.Label(top_frame, text="End (x,y):").grid(row=0, column=3, padx=5)
end_x = ttk.Entry(top_frame, width=5); end_x.insert(0, "550")
end_y = ttk.Entry(top_frame, width=5); end_y.insert(0, "350")
end_x.grid(row=0, column=4); end_y.grid(row=0, column=5)

ttk.Label(top_frame, text="Obstacles:").grid(row=0, column=6, padx=5)
num_obs = ttk.Entry(top_frame, width=5); num_obs.insert(0, "3")
num_obs.grid(row=0, column=7)

# status labels
cost_lbl = ttk.Label(bottom_frame, text="Cost: -")
cost_lbl.grid(row=0, column=0, padx=10)
lambda_lbl = ttk.Label(bottom_frame, text="λ_avg: -")
lambda_lbl.grid(row=0, column=1, padx=10)

# ---------- CALLBACKS ----------
def run_optimization():
    try:
        s = (int(start_x.get()), int(start_y.get()))
        e = (int(end_x.get()), int(end_y.get()))
        n = int(num_obs.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric inputs.")
        return

    obstacles = generate_obstacles(n)
    draw_environment(canvas, s, e, obstacles)
    path, cost, lam = lagrangian_optimizer(s, e, obstacles)
    draw_path(canvas, s, e, path)
    cost_lbl.config(text=f"Cost: {cost:.2f}")
    lambda_lbl.config(text=f"λ_avg: {lam:.3f}")

def predict_path():
    # Placeholder for Member 2 integration
    messagebox.showinfo("Predict Path (ML)",
                        "This button will call the ML model after integration.")

# buttons
opt_btn = ttk.Button(top_frame, text="Run Optimization", command=run_optimization)
opt_btn.grid(row=0, column=8, padx=10)

pred_btn = ttk.Button(top_frame, text="Predict Path (ML)", command=predict_path)
pred_btn.grid(row=0, column=9, padx=10)

root.mainloop()
