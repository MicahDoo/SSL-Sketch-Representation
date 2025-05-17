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
# from IPython.display import HTML, display # Keep commented for non-Jupyter script

# --- Script Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_FROM_SCRIPT = os.path.dirname(SCRIPT_DIR) 

sys.path.append(PROJECT_ROOT_FROM_SCRIPT) 
from data.dataset import SketchDataset

# --- Configuration ---
PROCESSED_DATA_DIR_ROOT = os.path.join(PROJECT_ROOT_FROM_SCRIPT, "processed_data")
CONFIG_FILENAME = "quickdraw_config.json" 

IMG_SIZE_TEST = 224
MAX_SEQ_LEN_TEST = 50

def denormalize_deltas(deltas_normalized, std_dev):
    """Denormalizes dx, dy if they were normalized."""
    if std_dev is not None and std_dev > 1e-6:
        deltas_np = np.array(deltas_normalized)
        denormalized_deltas = deltas_np.copy()
        denormalized_deltas[:, :2] *= std_dev
        return denormalized_deltas
    return np.array(deltas_normalized)

def get_drawn_segments_from_sequence(vector_sequence_np_denormalized):
    """
    Reconstructs drawn line segments from a denormalized delta sequence.
    The first point with p0=1 establishes the start, no line drawn *to* it from origin.
    Y-coordinates will be flipped (-y) to make y-axis point upwards for plotting.
    """
    current_x, current_y = 0.0, 0.0
    drawn_segments = []
    pen_is_at_first_drawn_point = False 

    for i in range(vector_sequence_np_denormalized.shape[0]):
        dx, dy, p0_is_pen_down, p1_is_pen_up, p2_is_eos = vector_sequence_np_denormalized[i]
        
        prev_x, prev_y = current_x, current_y 
        current_x += dx
        current_y += dy # Keep original y for logic, flip only for plotting list
        
        if p0_is_pen_down > 0.5: 
            if not pen_is_at_first_drawn_point:
                pen_is_at_first_drawn_point = True 
            else:
                # Flip y-coordinates for plotting
                drawn_segments.append(((prev_x, -prev_y), (current_x, -current_y)))
        else: 
            pen_is_at_first_drawn_point = False 
            
        if p2_is_eos > 0.5: 
            break
    return drawn_segments

def plot_colored_sequence(seq_normalized, std_dev=None, figsize=6):
    """
    Static plot of drawn segments with a reversed rainbow colormap.
    Y-coordinates are flipped for conventional display.
    """
    raw_denormalized = denormalize_deltas(seq_normalized, std_dev)
    segments_to_plot = get_drawn_segments_from_sequence(raw_denormalized) # Segments now have y flipped

    if not segments_to_plot:
        print("No line segments to plot for vector sequence.")
        if raw_denormalized.shape[0] == 1 and \
           raw_denormalized[0,2] > 0.5 and \
           raw_denormalized[0,4] > 0.5:
            point_x = raw_denormalized[0, 0] 
            point_y = -raw_denormalized[0, 1] # Flip y for single point
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            ax.plot(point_x, point_y, 'ko', markersize=5) 
            ax.set_title("Single Drawn Point")
            ax.set_aspect('equal'); ax.axis('off') 
            ax.set_xlim(point_x - 10, point_x + 10); ax.set_ylim(point_y - 10, point_y + 10)
            plt.show()
        return
        
    N = len(segments_to_plot)
    cmap = plt.get_cmap('rainbow_r', N if N > 0 else 1) # Reversed Rainbow

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    all_x, all_y = [], []
    for i, (start, end) in enumerate(segments_to_plot):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=cmap(i / (N -1) if N > 1 else 0.5 ), linewidth=2) 
        all_x.extend([start[0], end[0]])
        all_y.extend([start[1], end[1]])

    ax.set_aspect('equal'); ax.axis('off') 
    if all_x and all_y: 
        padding_x = (max(all_x) - min(all_x)) * 0.05 + 5 
        padding_y = (max(all_y) - min(all_y)) * 0.05 + 5
        ax.set_xlim(min(all_x) - padding_x, max(all_x) + padding_x)
        ax.set_ylim(min(all_y) - padding_y, max(all_y) + padding_y)

    if N > 0 :
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N-1))
        sm.set_array([]) 
        fig.colorbar(sm, ax=ax, ticks=[0, N//2, N-1] if N > 1 else [0]) \
           .set_ticklabels(['start','mid','end'] if N > 1 else ['segment'])
    ax.set_title("Vector Sketch Segments (Y-flipped, Rainbow_r)")
    plt.show()

def animate_sequence(seq_normalized, std_dev=None, interval=100, figsize=5):
    """
    Animated replay of drawn segments with a reversed rainbow colormap.
    Y-coordinates are flipped for conventional display.
    """
    raw_denormalized = denormalize_deltas(seq_normalized, std_dev)
    segments_to_animate = get_drawn_segments_from_sequence(raw_denormalized) # Segments now have y flipped

    if not segments_to_animate:
        print("No segments to animate.")
        return None 

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect('equal'); ax.axis('off') 
    
    all_x_coords_anim = [s[0] for s, e in segments_to_animate] + [e[0] for s, e in segments_to_animate]
    all_y_coords_anim = [s[1] for s, e in segments_to_animate] + [e[1] for s, e in segments_to_animate]
    if all_x_coords_anim and all_y_coords_anim:
        padding_x = (max(all_x_coords_anim) - min(all_x_coords_anim)) * 0.1 + 5 
        padding_y = (max(all_y_coords_anim) - min(all_y_coords_anim)) * 0.1 + 5
        ax.set_xlim(min(all_x_coords_anim) - padding_x, max(all_x_coords_anim) + padding_x) 
        ax.set_ylim(min(all_y_coords_anim) - padding_y, max(all_y_coords_anim) + padding_y)
    else: 
        ax.set_xlim(-10,10); ax.set_ylim(-10,10)

    lines = [] 
    cmap_anim = plt.get_cmap('rainbow_r', len(segments_to_animate) if len(segments_to_animate) > 0 else 1) # Reversed Rainbow
    for i in range(len(segments_to_animate)):
        line, = ax.plot([], [], '-', color=cmap_anim(i / (len(segments_to_animate)-1) if len(segments_to_animate) > 1 else 0.5), lw=2)
        lines.append(line)

    def init():
        for line in lines: line.set_data([], [])
        return lines

    def update(frame): 
        if frame < len(segments_to_animate):
            s, e = segments_to_animate[frame]
            lines[frame].set_data([s[0], e[0]], [s[1], e[1]])
        return lines

    ani = FuncAnimation(fig, update, frames=len(segments_to_animate), init_func=init,
                        interval=interval, blit=False, repeat=False) 
    plt.show() 
    return ani

def main():
    print("--- Testing SketchDataset with Color-Coded & Animated Renderings ---")
    print(f"DEBUG: Using PROCESSED_DATA_DIR_ROOT: {os.path.abspath(PROCESSED_DATA_DIR_ROOT)}")

    quickdraw_vector_dir = os.path.join(PROCESSED_DATA_DIR_ROOT, "quickdraw_vector")
    config_path = os.path.join(quickdraw_vector_dir, CONFIG_FILENAME)
    
    print(f"Attempting to load config for test script from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: '{config_path}'")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    std_dev_for_denorm = config.get("quickdraw_std_dev")
    category_map = config.get("category_map", {})
    label_to_category = {v: k for k, v in category_map.items()}

    split_to_test = "val"
    
    print(f"Initializing SketchDataset with dataset_path='{PROCESSED_DATA_DIR_ROOT}', dataset_name='quickdraw'")
    dataset = SketchDataset(
        dataset_path=PROCESSED_DATA_DIR_ROOT, 
        dataset_name="quickdraw",             
        split=split_to_test,
        max_seq_len=MAX_SEQ_LEN_TEST,
        image_size=IMG_SIZE_TEST,
        config_filename=CONFIG_FILENAME 
    )

    if len(dataset) == 0:
        print(f"Dataset for split '{split_to_test}' is empty. Check paths and preprocessing.")
        return

    num_samples = min(3, len(dataset)) 
    if num_samples == 0:
        print("No samples to show.")
        return
        
    sample_indices = random.sample(range(len(dataset)), num_samples)
    print(f"\n--- Showing {num_samples} random samples ---")

    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i+1} (index {idx}):")
        try:
            sample = dataset[idx]
            raster = sample["raster_image"]
            label_idx = sample["label"].item()
            cat = label_to_category.get(label_idx, f"Unknown_{label_idx}")

            print(f"  Category: {cat}, Raster shape: {raster.shape}")

            plt.figure(figsize=(4,4))
            plt.imshow(raster.squeeze(0).cpu().numpy(), cmap='gray_r') 
            plt.title(f"Raster '{cat}'")
            plt.axis('off')
            plt.show()

            if idx < len(dataset.processed_sequences):
                full_seq_normalized = dataset.processed_sequences[idx]
                plot_colored_sequence(full_seq_normalized, std_dev=std_dev_for_denorm)
                
                print("  Attempting to generate animation (this might take a moment)...")
                anim = animate_sequence(full_seq_normalized, std_dev=std_dev_for_denorm)
                if anim is None:
                    print("  Animation generation failed or skipped (no segments).")
            else:
                print("  Could not retrieve original vector sequence for plotting.")
        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            traceback.print_exc()

    print("\n--- Test script finished ---")

if __name__ == "__main__":
    if SketchDataset is None: 
        print("Exiting: SketchDataset class not imported. Check sys.path and file location.")
    else:
        main()
