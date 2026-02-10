import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_controls(img, controls):
    """Draw all control points on the image."""
    result = img.copy()
    for ctrl in controls:
        x, y = ctrl.x, ctrl.y
        color = (0, 50, 250) if ctrl.label != "0" else (255, 100, 0)
        cv2.circle(result, (x, y), 10, color, 2)
        cv2.circle(result, (x, y), 2, color, -1)
        if ctrl.label:
            cv2.putText(result, ctrl.label, (x + 12, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return result


def draw_path(img, path, color=(100, 0, 255), thickness=2):
    """Draw a path on the image with a single color."""
    result = img.copy()
    controls = path.controls
    for i in range(1, len(controls)):
        cv2.line(result, (controls[i - 1].x, controls[i - 1].y),
                 (controls[i].x, controls[i].y), color, thickness)
    return result


def draw_best_path(img, path):
    """Draw the best path with a green-to-red color gradient."""
    result = img.copy()
    controls = path.controls
    n = len(controls)
    if n < 2:
        return result
    step = 255 / max(n - 1, 1)
    r, g = 0, 255
    for i in range(1, n):
        cv2.line(result, (controls[i - 1].x, controls[i - 1].y),
                 (controls[i].x, controls[i].y),
                 (0, int(g), int(r)), 5)
        r = min(255, r + step)
        g = max(0, g - step)
    return result


def plot_controls_3d(controls):
    """Plot controls in 3D (easting, northing, elevation)."""
    if not controls:
        print("No controls to plot.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = [c.easting for c in controls if c.easting is not None]
    ys = [c.northing for c in controls if c.northing is not None]
    zs = [c.elevation if c.elevation is not None else 0 for c in controls if c.easting is not None]
    labels = [c.label for c in controls if c.easting is not None]

    ax.scatter(xs, ys, zs, c="r", marker="o", s=50)
    for i, label in enumerate(labels):
        if label:
            ax.text(xs[i], ys[i], zs[i], label)

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Elevation (m)")
    ax.set_title("Control Points in 3D")
    plt.tight_layout()
    plt.show()


def show_result(img, path, window_name="Best Path"):
    """Display the best path result in an OpenCV window."""
    result = draw_controls(img, path.controls)
    result = draw_best_path(result, path)
    cv2.imshow(window_name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
