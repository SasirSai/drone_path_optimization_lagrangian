from tkinter import Canvas

def draw_environment(canvas: Canvas, start, end, obstacles):
    """Draw static elements: start, end, obstacles."""
    canvas.delete("all")
    canvas.create_oval(start[0]-5, start[1]-5, start[0]+5, start[1]+5,
                       fill="green", outline="")
    canvas.create_oval(end[0]-5, end[1]-5, end[0]+5, end[1]+5,
                       fill="red", outline="")
    for (center, r) in obstacles:
        x, y = center
        canvas.create_oval(x-r, y-r, x+r, y+r, outline="gray", width=2)

def draw_path(canvas: Canvas, start, end, path_points, color="blue"):
    """Draw optimized path on canvas."""
    pts = [start] + path_points.tolist() + [end]
    for i in range(len(pts)-1):
        x1, y1 = pts[i]
        x2, y2 = pts[i+1]
        canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
