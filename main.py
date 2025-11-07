import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from core.optimizer import generate_obstacles, lagrangian_optimizer
from gui.visualizer import draw_environment
from ml.model_predict import predict_path_cost
from ml.model_train import train_and_save_model


# ============================================================
#           MODERN UI STYLE
# ============================================================
def apply_modern_style(root):
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TButton",
                    font=("Segoe UI", 11, "bold"),
                    padding=6,
                    foreground="white",
                    background="#3A7FF6",
                    borderwidth=0)
    style.map("TButton",
              background=[("active", "#1E5FCC")])

    style.configure("TLabel",
                    font=("Segoe UI", 11),
                    background="#1e1e1e",
                    foreground="white")

    style.configure("TEntry",
                    font=("Segoe UI", 11),
                    padding=5)


# ============================================================
#                DRONE ANIMATION ENGINE
# ============================================================
def animate_drone(canvas, path, speed=3):
    """ Smooth drone animation across the optimized path. """
    if path is None or len(path) < 2:
        return

    # visual drone parameters
    drone_radius = 6
    drone_color = "#00FFAA"

    # remove any existing drone
    canvas.delete("drone")

    # drone start
    x, y = path[0]
    drone = canvas.create_oval(x-drone_radius, y-drone_radius,
                               x+drone_radius, y+drone_radius,
                               fill=drone_color, outline="", tag="drone")

    # animate along segments
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]

        steps = int(max(abs(x2-x1), abs(y2-y1)) / speed)
        steps = max(steps, 1)

        for j in range(steps):
            t = j / steps
            xt = x1 + (x2 - x1) * t
            yt = y1 + (y2 - y1) * t

            canvas.coords(drone,
                          xt - drone_radius, yt - drone_radius,
                          xt + drone_radius, yt + drone_radius)
            canvas.update()
            canvas.after(5)  # controls animation smoothness

    return


# ============================================================
#                      GUI SETUP
# ============================================================
root = tk.Tk()
root.title("Drone Path Optimization – Lagrangian Method")
root.geometry("950x600")
root.config(bg="#1e1e1e")
apply_modern_style(root)

# Layout Frames
left_panel = tk.Frame(root, bg="#1e1e1e", width=250)
left_panel.pack(side="left", fill="y")

canvas_frame = tk.Frame(root, bg="#1e1e1e")
canvas_frame.pack(side="right", expand=True)

canvas = tk.Canvas(canvas_frame, width=650, height=500,
                   bg="#161616", highlightthickness=0)
canvas.pack(padx=20, pady=20)


# ============================================================
#             INPUT FORM ELEMENTS
# ============================================================
def labeled_entry(parent, text, default):
    frame = tk.Frame(parent, bg="#1e1e1e")
    frame.pack(pady=4)
    tk.Label(frame, text=text, bg="#1e1e1e", fg="white").pack(anchor="w")
    ent = ttk.Entry(frame)
    ent.insert(0, default)
    ent.pack(fill="x")
    return ent


tk.Label(left_panel, text="✈ Drone Optimizer", font=("Segoe UI", 16, "bold"),
         bg="#1e1e1e", fg="white").pack(pady=15)

start_x = labeled_entry(left_panel, "Start X:", "50")
start_y = labeled_entry(left_panel, "Start Y:", "50")
end_x = labeled_entry(left_panel, "End X:", "550")
end_y = labeled_entry(left_panel, "End Y:", "350")
num_obs = labeled_entry(left_panel, "Number of Obstacles:", "4")


# STATUS PANEL
status_label = tk.Label(left_panel, text="Status: Idle",
                        bg="#1e1e1e", fg="#00FFAA",
                        font=("Segoe UI", 11, "italic"))
status_label.pack(pady=15)


def set_status(text):
    status_label.config(text=f"Status: {text}")
    status_label.update()


# COST INFO
cost_lbl = tk.Label(left_panel, text="Cost: -", bg="#1e1e1e", fg="white",
                    font=("Segoe UI", 12))
cost_lbl.pack()

lambda_lbl = tk.Label(left_panel, text="λ_avg: -", bg="#1e1e1e", fg="white",
                      font=("Segoe UI", 12))
lambda_lbl.pack(pady=5)


# ============================================================
#                  CALLBACK FUNCTIONS
# ============================================================
def run_optimization():
    try:
        s = (int(start_x.get()), int(start_y.get()))
        e = (int(end_x.get()), int(end_y.get()))
        n = int(num_obs.get())
    except:
        messagebox.showerror("Input Error", "Enter valid integers!")
        return

    set_status("Generating obstacles...")
    obstacles = generate_obstacles(n, start=s, end=e)

    # Draw obstacles
    draw_environment(canvas, s, e, obstacles)

    set_status("Running optimization...")
    path, cost, lam = lagrangian_optimizer(s, e, obstacles)

    # Draw optimized path
    canvas.delete("path")
    for i in range(len(path)-1):
        canvas.create_line(path[i][0], path[i][1],
                           path[i+1][0], path[i+1][1],
                           fill="#00E5FF", width=3, smooth=True, tag="path")

    # Update UI
    cost_lbl.config(text=f"Cost: {cost:.2f}")
    lambda_lbl.config(text=f"λ_avg: {lam:.3f}")
    set_status("Path ready. Animating drone...")

    # Animate drone
    animate_drone(canvas, path)

    set_status("Completed!")


def predict_path():
    try:
        s = (int(start_x.get()), int(start_y.get()))
        e = (int(end_x.get()), int(end_y.get()))
        n = int(num_obs.get())
    except:
        messagebox.showerror("Input Error", "Invalid input!")
        return

    lam_text = lambda_lbl.cget("text").replace("λ_avg:", "").strip()
    lam_val = float(lam_text) if lam_text != "-" else 25.0

    try:
        pred = predict_path_cost(s, e, n, lam_val)
        messagebox.showinfo("Predicted Path Cost", f"Predicted Cost ≈ {pred:.2f}")
        set_status("Prediction completed.")
    except Exception as ex:
        messagebox.showerror("Prediction Error", str(ex))


def train_ml_now():
    try:
        set_status("Training ML model...")
        train_and_save_model()
        messagebox.showinfo("ML Training", "Model trained successfully!")
        set_status("ML Model Updated.")
    except Exception as ex:
        messagebox.showerror("ML Training Error", str(ex))


# ============================================================
#                     BUTTONS
# ============================================================
def modern_button(text, cmd):
    b = ttk.Button(left_panel, text=text, command=cmd)
    b.pack(fill="x", pady=5)
    return b


modern_button("Run Optimization", run_optimization)
modern_button("Predict Cost (ML)", predict_path)
modern_button("Train / Update ML", train_ml_now)


# ============================================================
root.mainloop()
