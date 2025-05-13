import os
import numpy as np
import json
import random
import requests
from tqdm import tqdm
import argparse

# --- Configuration ---
BASE_DOWNLOAD_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
DEFAULT_CATEGORIES = [
    "cat", "dog", "apple", "car", "tree", "house", "bicycle", "bird", "face", "airplane"
]

RAW_DOWNLOAD_DIR_ROOT = os.path.join("..", "downloaded_data")
PROCESSED_DATA_DIR_ROOT = os.path.join("..", "processed_data")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VAL_RATIO

CONFIG_FILENAME = "quickdraw_config.json"

def download_category_file(category_name, target_dir):
    """Downloads a single QuickDraw category .npy file if it doesn't exist."""
    raw_cat_dir = os.path.join(target_dir, "quickdraw_raw")
    os.makedirs(raw_cat_dir, exist_ok=True)
    
    file_name = f"{category_name}.npy"
    url = f"{BASE_DOWNLOAD_URL}{file_name}"
    target_path = os.path.join(raw_cat_dir, file_name)

    if os.path.exists(target_path):
        print(f"File '{file_name}' already exists in '{raw_cat_dir}'. Skipping download.")
        return target_path, True

    print(f"Downloading '{file_name}' from '{url}'...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f, tqdm(
            desc=file_name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024*4):
                size = f.write(data)
                bar.update(size)
        print(f"Successfully downloaded '{file_name}' to '{raw_cat_dir}'.")
        return target_path, True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(target_path): os.remove(target_path)
        return target_path, False

def convert_raw_sketch_to_delta_sequence(raw_sketch_strokes):
    """
    Converts a raw QuickDraw sketch (list of absolute strokes) to a 
    (dx, dy, p0, p1, p2) sequence.
    p0: pen_down (stroke continues)
    p1: pen_up (end of a stroke, pen lifts)
    p2: end_of_sketch (last point of the sketch)
    Returns a NumPy array of shape (num_points, 5), dtype=np.float32.
    Deltas are NOT normalized at this stage.
    """
    points = []
    last_x, last_y = 0, 0
    num_strokes = len(raw_sketch_strokes)

    if num_strokes == 0:
        return np.array([], dtype=np.float32).reshape(0, 5)

    for stroke_idx, stroke in enumerate(raw_sketch_strokes):
        stroke_xs, stroke_ys = stroke[0], stroke[1]
        if len(stroke_xs) == 0:
            continue

        # Move to the start of the current stroke (implicit pen_up from previous, or from 0,0)
        # The (dx, dy) for this point is from (last_x, last_y) to (stroke_xs[0], stroke_ys[0])
        # Pen state: If not the first stroke, this is a pen_up (p1=1).
        #            If it's the first point of the first stroke, this dx,dy is the first move.
        
        delta_x = stroke_xs[0] - last_x
        delta_y = stroke_ys[0] - last_y
        
        # For the very first point of the sketch, it's a "pen_down" to start.
        # For subsequent strokes, the point that starts the stroke comes after a "pen_up".
        if stroke_idx > 0: # This point signifies the end of a pen-up travel
            points.append([delta_x, delta_y, 0, 1, 0]) # p1=1 (pen_up completed, ready for new stroke)
        else: # First stroke's first point (no preceding pen-up in the sequence)
            # This first delta is recorded, and it's immediately a pen-down.
            # It's effectively a move from (0,0) with pen down.
             pass # The first actual point data is handled below as a pen-down


        # Current stroke points (all pen_down, p0=1, unless it's the very last point of the sketch)
        # The first point of any stroke sequence has dx=0, dy=0 relative to the pen landing,
        # unless it's the very first point of the sketch.
        
        # Special handling for the first actual drawing point of a stroke
        if stroke_idx == 0: # First stroke
            current_dx = stroke_xs[0] - last_x # Delta from (0,0) for the very first point
            current_dy = stroke_ys[0] - last_y
        else: # Subsequent strokes start with (0,0) delta after the pen-up move
            current_dx = 0 
            current_dy = 0
        
        # Update last_x, last_y to the start of this stroke
        last_x, last_y = stroke_xs[0], stroke_ys[0]

        # Add the first point of the current stroke
        is_last_point_overall = (stroke_idx == num_strokes - 1) and (len(stroke_xs) == 1)
        pen_state = [0,0,1] if is_last_point_overall else [1,0,0]
        points.append([current_dx, current_dy] + pen_state)


        # Add subsequent points in the current stroke
        for i in range(1, len(stroke_xs)):
            delta_x = stroke_xs[i] - last_x
            delta_y = stroke_ys[i] - last_y
            last_x, last_y = stroke_xs[i], stroke_ys[i]
            
            is_last_point_overall = (stroke_idx == num_strokes - 1) and (i == len(stroke_xs) - 1)
            pen_state = [0,0,1] if is_last_point_overall else [1,0,0]
            points.append([delta_x, delta_y] + pen_state)
            
    if not points: # Should not happen if num_strokes > 0 and strokes are not empty
        return np.array([], dtype=np.float32).reshape(0, 5)
        
    return np.array(points, dtype=np.float32)


def main(categories_to_process, raw_data_dir, processed_data_dir,
         train_r, val_r):
    
    print(f"--- Starting QuickDraw Dataset Preprocessing (Heavy Mode) ---")
    # ... (directory creation and initial messages - same as before) ...
    quickdraw_processed_base = os.path.join(processed_data_dir, "quickdraw")
    os.makedirs(quickdraw_processed_base, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(quickdraw_processed_base, split), exist_ok=True)

    # Store all raw sketches split by train/val/test for pass 2
    all_splits_raw_sketches = {"train": [], "val": [], "test": []}
    category_map = {}
    
    categories_to_process.sort()
    for cat_idx, category_name in enumerate(categories_to_process):
        print(f"\nProcessing category: '{category_name}' ({cat_idx + 1}/{len(categories_to_process)})...")
        category_map[category_name] = cat_idx

        raw_npy_path, downloaded = download_category_file(category_name, raw_data_dir)
        if not downloaded and not os.path.exists(raw_npy_path):
            print(f"Failed to obtain raw file for '{category_name}'. Skipping.")
            continue
        
        try:
            sketches_raw_strokes = np.load(raw_npy_path, encoding='latin1', allow_pickle=True)
        except Exception as e:
            print(f"Could not load raw sketches from '{raw_npy_path}': {e}. Skipping category.")
            continue

        if len(sketches_raw_strokes) == 0: continue
            
        random.shuffle(sketches_raw_strokes)
        num_total = len(sketches_raw_strokes)
        num_train = int(train_r * num_total)
        num_val = int(val_r * num_total)

        # Store raw sketches with their category name for Pass 1 (std_dev) and Pass 2 (processing)
        all_splits_raw_sketches["train"].extend([(category_name, s) for s in sketches_raw_strokes[:num_train]])
        all_splits_raw_sketches["val"].extend([(category_name, s) for s in sketches_raw_strokes[num_train : num_train + num_val]])
        all_splits_raw_sketches["test"].extend([(category_name, s) for s in sketches_raw_strokes[num_train + num_val :]])

    # --- Pass 1: Calculate std_dev from training sketches' UNNORMALIZED deltas ---
    print("\n--- Pass 1: Calculating std_dev from training set deltas ---")
    all_training_dx_dy_unnormalized = []
    for category_name, raw_sketch in all_splits_raw_sketches["train"]:
        delta_sequence_unnormalized = convert_raw_sketch_to_delta_sequence(raw_sketch)
        if delta_sequence_unnormalized.shape[0] > 0:
            all_training_dx_dy_unnormalized.extend(delta_sequence_unnormalized[:, 0].tolist()) # All dx
            all_training_dx_dy_unnormalized.extend(delta_sequence_unnormalized[:, 1].tolist()) # All dy
    
    if not all_training_dx_dy_unnormalized:
        print("Warning: No training deltas collected. Cannot calculate std_dev. Setting to 1.0.")
        std_dev = 1.0
    else:
        std_dev = np.std(all_training_dx_dy_unnormalized)
        if std_dev < 1e-6:
            print(f"Warning: Calculated std_dev is very small ({std_dev}). Setting to 1.0.")
            std_dev = 1.0
    print(f"Calculated global std_dev from training deltas: {std_dev:.4f}")

    # --- Pass 2: Convert all sketches, normalize, and save splits ---
    print("\n--- Pass 2: Converting, Normalizing, and Saving all splits ---")
    for split_name, list_of_cat_sketch_pairs in all_splits_raw_sketches.items():
        # Group sketches by category for saving
        sketches_by_category_for_split = {}
        for category_name, raw_sketch in list_of_cat_sketch_pairs:
            if category_name not in sketches_by_category_for_split:
                sketches_by_category_for_split[category_name] = []
            
            delta_sequence_unnormalized = convert_raw_sketch_to_delta_sequence(raw_sketch)
            if delta_sequence_unnormalized.shape[0] == 0:
                continue

            # Normalize dx, dy
            normalized_sequence = delta_sequence_unnormalized.copy()
            if std_dev > 1e-6 : # Avoid division by zero/small number
                 normalized_sequence[:, :2] /= std_dev
            
            sketches_by_category_for_split[category_name].append(normalized_sequence)
        
        # Save processed sketches for this split
        for category_name, processed_sequences_list in sketches_by_category_for_split.items():
            if processed_sequences_list:
                output_filename = f"{category_name}.npy"
                output_path = os.path.join(quickdraw_processed_base, split_name, output_filename)
                # Save as a NumPy array of objects, where each object is a processed sequence array
                np.save(output_path, np.array(processed_sequences_list, dtype=object))
                print(f"  Saved {len(processed_sequences_list)} processed sketches for '{category_name}' to '{split_name}' split.")

    # Save config
    config_data = {
        "quickdraw_std_dev": float(std_dev),
        "category_map": category_map,
        "categories_processed": categories_to_process,
        "train_ratio": train_r, "val_ratio": val_r,
        "data_path_structure_note": "Sketches are PROCESSED (dx,dy,p_states), NORMALIZED sequences, split by category."
    }
    config_path = os.path.join(quickdraw_processed_base, CONFIG_FILENAME)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"Saved preprocessing config to '{os.path.abspath(config_path)}'")
    print("\n--- QuickDraw Dataset Preprocessing (Heavy Mode) Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, convert, normalize, and split QuickDraw dataset.")
    # ... (argparse setup same as before) ...
    parser.add_argument(
        "--categories", nargs="+", default=DEFAULT_CATEGORIES,
        help=f"List of QuickDraw categories to process. Default: {len(DEFAULT_CATEGORIES)} categories."
    )
    parser.add_argument(
        "--raw_dir", default=RAW_DOWNLOAD_DIR_ROOT,
        help="Directory to store/check for raw downloaded .npy files."
    )
    parser.add_argument(
        "--processed_dir", default=PROCESSED_DATA_DIR_ROOT,
        help="Root directory to store processed (split) data and config."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=TRAIN_RATIO, help="Proportion of data for training."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=VAL_RATIO, help="Proportion of data for validation."
    )
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("Sum of train_ratio and val_ratio must be less than 1.0.")

    main(args.categories, args.raw_dir, args.processed_dir, 
         args.train_ratio, args.val_ratio)