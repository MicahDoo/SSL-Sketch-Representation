import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Adjust path to import SketchDataset from ../data/dataset.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.append(project_root)
from data.dataset import SketchDataset 

# --- Configuration ---
PROCESSED_DATA_DIR_ROOT = os.path.join("..", "processed_data")
CONFIG_FILENAME = "quickdraw_config.json" # Expected in processed_data/quickdraw/

IMG_SIZE_TEST = 64 
MAX_SEQ_LEN_TEST = 50 

def denormalize_deltas(deltas_normalized, std_dev):
    """Denormalizes dx, dy if they were normalized."""
    if std_dev is not None and std_dev > 1e-6: # Check std_dev is a meaningful number
        # Ensure deltas_normalized is a NumPy array for broadcasting if needed
        deltas_np = np.array(deltas_normalized)
        return deltas_np * std_dev
    return np.array(deltas_normalized)


def plot_vector_sketch_from_processed_sequence(processed_sequence_tensor, title="Vector Sketch", std_dev_for_denorm=None):
    """
    Plots a sketch from its preprocessed (dx, dy, p0, p1, p2) sequence.
    processed_sequence_tensor: Tensor of shape (seq_len, 5), dx/dy might be normalized.
    std_dev_for_denorm: The std_dev used during preprocessing, to denormalize for plotting.
    """
    if not isinstance(processed_sequence_tensor, np.ndarray):
        sketch_data_padded = processed_sequence_tensor.cpu().numpy()
    else:
        sketch_data_padded = processed_sequence_tensor

    # Remove padding for plotting - find first row where all 5 features are effectively zero
    # or rely on attention mask if it were passed.
    # A simple way: if sum of absolute values of a row is very small, assume padding.
    non_padding_indices = np.where(np.sum(np.abs(sketch_data_padded), axis=1) > 1e-5)[0]
    if len(non_padding_indices) > 0:
        actual_len = non_padding_indices[-1] + 1
        sketch_data = sketch_data_padded[:actual_len]
    else: # All padding or empty
        sketch_data = sketch_data_padded[0:0] # Empty slice with correct dimensions

    if sketch_data.shape[0] == 0:
        print(f"Plotting: No non-padding points found in sketch for '{title}'.")
        # Create an empty plot for consistency if desired
        plt.figure(figsize=(6, 6))
        plt.title(f"{title} (Empty or all padding)")
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.show()
        return

    plt.figure(figsize=(6, 6))
    current_x, current_y = 0.0, 0.0
    
    # Denormalize the dx, dy columns for plotting if std_dev is provided
    plot_deltas = sketch_data[:, :2].copy() # Get dx, dy
    if std_dev_for_denorm is not None:
        plot_deltas = denormalize_deltas(plot_deltas, std_dev_for_denorm)

    for i in range(sketch_data.shape[0]):
        # dx, dy are from plot_deltas (potentially denormalized)
        # p_states are from original sketch_data
        plot_dx, plot_dy = plot_deltas[i]
        _, _, p_down, p_up, p_eos = sketch_data[i] 

        next_x, next_y = current_x + plot_dx, current_y + plot_dy
        
        if p_down > 0.5: # Pen is down
            plt.plot([current_x, next_x], [current_y, next_y], 'k-')
        
        current_x, current_y = next_x, next_y
        
        if p_eos > 0.5: break
            
    plt.title(title)
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.show()


def plot_raster_image(raster_tensor, title="Raster Image"):
    if raster_tensor.is_cuda: raster_tensor = raster_tensor.cpu()
    img_data = raster_tensor.squeeze(0).numpy()
    plt.figure(figsize=(4,4))
    plt.imshow(img_data, cmap='gray_r')
    plt.title(title)
    plt.show()

def main():
    print("--- Testing SketchDataset with HEAVILY Preprocessed QuickDraw Data ---")

    quickdraw_processed_dir = os.path.join(PROCESSED_DATA_DIR_ROOT, "quickdraw")
    config_path = os.path.join(quickdraw_processed_dir, CONFIG_FILENAME)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: '{os.path.abspath(config_path)}'")
        print("Please run 'preprocess_dataset_quickdraw.py' (heavy mode) first.")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}"); return
        
    std_dev_for_denorm = config.get("quickdraw_std_dev") # For plotting denormalized
    category_map = config.get("category_map", {})
    label_to_category = {v: k for k, v in category_map.items()}

    if std_dev_for_denorm is None:
        print("Warning: 'quickdraw_std_dev' not found in config. Plotting will use normalized deltas.")

    split_to_test = "val"
    print(f"\n--- Initializing SketchDataset for '{split_to_test}' split ---")
    
    try:
        dataset = SketchDataset(
            dataset_path=PROCESSED_DATA_DIR_ROOT, 
            dataset_name="quickdraw",
            split=split_to_test,
            max_seq_len=MAX_SEQ_LEN_TEST,
            image_size=IMG_SIZE_TEST
            # No std_dev passed to SketchDataset constructor, as data is pre-normalized
        )
    except Exception as e:
        print(f"Error initializing SketchDataset: {e}"); traceback.print_exc(); return

    print(f"Dataset length for '{split_to_test}': {len(dataset)}")
    if hasattr(dataset, 'category_map') and dataset.category_map:
         print(f"SketchDataset internal category map: {dataset.category_map}")

    if len(dataset) == 0: print(f"Dataset for '{split_to_test}' is empty."); return

    num_samples_to_show = min(3, len(dataset))
    print(f"\n--- Showing first {num_samples_to_show} samples from '{split_to_test}' split ---")
    
    for i in range(num_samples_to_show):
        print(f"\nSample {i}:")
        try:
            sample = dataset[i]
            vec_sketch = sample["vector_sketch"]
            raster_img = sample["raster_image"]
            attn_mask = sample["attention_mask"]
            label_idx = sample["label"].item()
            category_name = label_to_category.get(label_idx, f"Unknown_{label_idx}")

            print(f"  Label: {label_idx} (Category: '{category_name}')")
            print(f"  Vector Sketch Shape (Padded/Truncated): {vec_sketch.shape}")
            active_points = torch.sum(attn_mask).item()
            print(f"  Attention Mask: Active points = {int(active_points)} / {MAX_SEQ_LEN_TEST}")
            # print(f"  Vector Sketch (first 3 points, normalized): \n{vec_sketch[:3, :]}")
            
            print(f"  Raster Image Shape: {raster_img.shape}")

            plot_title_vec = f"Sample {i}: Vector '{category_name}' (Normalized, then Denorm for Plot)"
            plot_vector_sketch_from_processed_sequence(vec_sketch, title=plot_title_vec, std_dev_for_denorm=std_dev_for_denorm)
            
            plot_title_raster = f"Sample {i}: Raster '{category_name}'"
            plot_raster_image(raster_img, title=plot_title_raster)

        except Exception as e:
            print(f"Error processing or plotting sample {i}: {e}"); traceback.print_exc()
            
    print("\n--- Test script finished ---")

if __name__ == "__main__":
    main()