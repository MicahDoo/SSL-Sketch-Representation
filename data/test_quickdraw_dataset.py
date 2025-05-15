import os
import sys
import json
import random
import traceback

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# Adjust path to import SketchDataset from ../data/dataset.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from data.dataset import SketchDataset

# --- Configuration ---
PROCESSED_DATA_DIR_ROOT = os.path.join("..", "processed_data")
CONFIG_FILENAME = "quickdraw_config.json"  # Expected in processed_data/quickdraw/

IMG_SIZE_TEST = 224
MAX_SEQ_LEN_TEST = 50

def denormalize_deltas(deltas_normalized, std_dev):
    """Denormalizes dx, dy if they were normalized."""
    if std_dev is not None and std_dev > 1e-6:
        return np.array(deltas_normalized) * std_dev
    return np.array(deltas_normalized)

def chaikin_smooth(points, iterations=2):
    """
    Applies Chaikin's corner-cutting algorithm to smooth a polyline.
    points: list of (x,y) tuples
    iterations: number of smoothing passes
    """
    pts = [np.array(p) for p in points]
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts

def plot_colored_sequence(seq, std_dev=None, figsize=6, smooth_iters=2):
    """
    Static rainbow plot (red→purple), disconnected strokes, smoothed.
    """
    raw = seq.cpu().numpy() if hasattr(seq, "cpu") else np.array(seq)
    # Build absolute points
    abs_pts = [(0.0, 0.0)]
    strokes = []
    current = []
    for dx, dy, p0, p1, p2 in raw:
        last = abs_pts[-1]
        nxt = (last[0] + (dx * std_dev if std_dev else dx),
               last[1] + (dy * std_dev if std_dev else dy))
        abs_pts.append(nxt)
        if p0 > 0.5:
            current.append(nxt)
        if p1 > 0.5 or p2 > 0.5:  # stroke ends
            if current:
                # include the starting point of stroke
                stroke = [abs_pts[-len(current)-1]] + current
                strokes.append(stroke)
            current = []

    # Apply smoothing per stroke and collect segments
    segments = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        sm_pts = chaikin_smooth(stroke, iterations=smooth_iters)
        for a, b in zip(sm_pts[:-1], sm_pts[1:]):
            segments.append((a, b))

    N = len(segments)
    cmap = plt.get_cmap('rainbow', N)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    for i, (start, end) in enumerate(segments):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=cmap(i), linewidth=2)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    sm = cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=0, vmax=N-1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, ticks=[0, N//2, N-1]) \
       .set_ticklabels(['start','mid','end'])
    ax.set_title("Smoothed Rainbow Stroke Order")
    plt.show()

def animate_sequence(seq, std_dev=None, interval=100, figsize=5, smooth_iters=2):
    """
    Animated rainbow replay, smoothed strokes.
    """
    raw = seq.cpu().numpy() if hasattr(seq, "cpu") else np.array(seq)
    abs_pts = [(0.0, 0.0)]
    strokes = []
    current = []
    for dx, dy, p0, p1, p2 in raw:
        last = abs_pts[-1]
        nxt = (last[0] + (dx * std_dev if std_dev else dx),
               last[1] + (dy * std_dev if std_dev else dy))
        abs_pts.append(nxt)
        if p0 > 0.5:
            current.append(nxt)
        if p1 > 0.5 or p2 > 0.5:
            if current:
                stroke = [abs_pts[-len(current)-1]] + current
                strokes.append(stroke)
            current = []

    segments = []
    for stroke in strokes:
        if len(stroke) < 2: continue
        sm_pts = chaikin_smooth(stroke, iterations=smooth_iters)
        for a, b in zip(sm_pts[:-1], sm_pts[1:]):
            segments.append((a, b))

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')

    def update(frame):
        s, e = segments[frame]
        ax.plot([s[0], e[0]], [s[1], e[1]], '-', color=plt.get_cmap('rainbow')(frame/len(segments)), lw=2)
    ani = FuncAnimation(fig, update, frames=len(segments),
                        interval=interval, blit=False)
    plt.show()
    return ani


def main():
    print("--- Testing SketchDataset with Color-Coded & Animated Renderings ---")

    quickdraw_processed_dir = os.path.join(PROCESSED_DATA_DIR_ROOT, "quickdraw")
    config_path = os.path.join(quickdraw_processed_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: '{config_path}'")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    std_dev_for_denorm = config.get("quickdraw_std_dev")
    category_map = config.get("category_map", {})
    label_to_category = {v: k for k, v in category_map.items()}

    split_to_test = "val"
    dataset = SketchDataset(
        dataset_path=PROCESSED_DATA_DIR_ROOT,
        dataset_name="quickdraw",
        split=split_to_test,
        max_seq_len=MAX_SEQ_LEN_TEST,
        image_size=IMG_SIZE_TEST
    )

    num_samples = min(5, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    print(f"\n--- Showing {num_samples} random samples ---")

    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i} (index {idx}):")
        sample = dataset[idx]
        raster = sample["raster_image"]
        label_idx = sample["label"].item()
        cat = label_to_category.get(label_idx, f"Unknown_{label_idx}")

        print(f"  Category: {cat}, Raster shape: {raster.shape}")

        # Show raster
        plt.figure(figsize=(4,4))
        plt.imshow(raster.squeeze(0).cpu().numpy(), cmap='gray_r')
        plt.title(f"Raster '{cat}'")
        plt.axis('off')
        plt.show()

        # Static color-coded
        full_seq = dataset.processed_sequences[idx]
        plot_colored_sequence(full_seq, std_dev=std_dev_for_denorm)

        # Animated replay
        anim = animate_sequence(full_seq, std_dev=std_dev_for_denorm)
        if anim is not None:
            display(anim)

    print("\n--- Test script finished ---")


if __name__ == "__main__":
    main()

