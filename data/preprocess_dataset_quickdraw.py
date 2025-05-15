import os
import json
import random
import argparse
import numpy as np
import requests
from tqdm import tqdm

# --- Configuration ---
BASE_DOWNLOAD_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
DEFAULT_CATEGORIES = [
    "cat", "dog", "apple", "car", "tree", "house", "bicycle", "bird", "face", "airplane"
]

RAW_DOWNLOAD_DIR_ROOT = os.path.join("..", "downloaded_data")
PROCESSED_DATA_DIR_ROOT = os.path.join("..", "processed_data")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

CONFIG_FILENAME = "quickdraw_config.json"

def download_category_file(category_name, target_dir):
    """Download a QuickDraw simplified .ndjson file if it doesn't exist."""
    raw_cat_dir = os.path.join(target_dir, "quickdraw_raw")
    os.makedirs(raw_cat_dir, exist_ok=True)
    
    file_name = f"{category_name}.ndjson"
    url = f"{BASE_DOWNLOAD_URL}{file_name}"
    target_path = os.path.join(raw_cat_dir, file_name)

    if os.path.exists(target_path):
        print(f"File '{file_name}' already exists in '{raw_cat_dir}'. Skipping download.")
        return target_path, True

    print(f"Downloading '{file_name}' from '{url}'...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f, tqdm(
            desc=file_name, total=int(response.headers.get('content-length', 0)),
            unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Successfully downloaded '{file_name}' to '{raw_cat_dir}'.")
        return target_path, True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return target_path, False

def load_simplified_drawings(ndjson_path, max_items=None):
    """
    Reads a .ndjson file and returns a list of drawings,
    where each drawing is a list of strokes [[xs],[ys],…].
    """
    drawings = []
    with open(ndjson_path, 'r') as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            try:
                obj = json.loads(line)
                drawings.append(obj['drawing'])
            except json.JSONDecodeError:
                continue
    return drawings

def convert_raw_sketch_to_delta_sequence(raw_sketch_strokes):
    """
    Converts a QuickDraw sketch (list of strokes) to a
    (dx, dy, p0, p1, p2) sequence.
    p0: pen_down (stroke continues)
    p1: pen_up (end of a stroke)
    p2: end_of_sketch
    Returns np.ndarray of shape (num_points, 5), dtype=float32.
    """
    points = []
    last_x, last_y = 0, 0
    num_strokes = len(raw_sketch_strokes)
    if num_strokes == 0:
        return np.zeros((0,5), dtype=np.float32)

    for stroke_idx, stroke in enumerate(raw_sketch_strokes):
        xs, ys = stroke
        if not xs:
            continue

        # Pen-up move to start of this stroke (except first stroke)
        dx0 = xs[0] - last_x
        dy0 = ys[0] - last_y
        if stroke_idx > 0:
            points.append([dx0, dy0, 0, 1, 0])
            dx0, dy0 = 0, 0  # reset delta for the first drawing point

        # First point of stroke
        is_last_point = (stroke_idx == num_strokes - 1) and (len(xs) == 1)
        p_state = [0,0,1] if is_last_point else [1,0,0]
        points.append([dx0, dy0] + p_state)
        last_x, last_y = xs[0], ys[0]

        # Remaining points
        for i in range(1, len(xs)):
            dx = xs[i] - last_x
            dy = ys[i] - last_y
            last_x, last_y = xs[i], ys[i]
            is_last_point = (stroke_idx == num_strokes - 1) and (i == len(xs) - 1)
            p_state = [0,0,1] if is_last_point else [1,0,0]
            points.append([dx, dy] + p_state)

    return np.array(points, dtype=np.float32)

def main(categories, raw_dir, processed_dir, train_r, val_r):
    print("--- Starting QuickDraw Preprocessing (Vector mode) ---")

    # Prepare directories
    proc_base = os.path.join(processed_dir, "quickdraw")
    os.makedirs(proc_base, exist_ok=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(proc_base, split), exist_ok=True)

    # Download & load all sketches, then split
    all_splits = {"train": [], "val": [], "test": []}
    category_map = {}
    categories = sorted(categories)

    for idx, cat in enumerate(categories):
        print(f"\nProcessing category '{cat}' ({idx+1}/{len(categories)})")
        category_map[cat] = idx

        ndjson_path, ok = download_category_file(cat, raw_dir)
        if not ok:
            print(f"  → Failed to download '{cat}', skipping.")
            continue

        drawings = load_simplified_drawings(ndjson_path)
        random.shuffle(drawings)
        n = len(drawings)
        n_train = int(train_r * n)
        n_val   = int(val_r   * n)

        all_splits["train"].extend((cat, d) for d in drawings[:n_train])
        all_splits["val"].extend(  (cat, d) for d in drawings[n_train:n_train+n_val])
        all_splits["test"].extend( (cat, d) for d in drawings[n_train+n_val:])

    # Pass 1: compute std_dev from train deltas
    print("\n--- Pass 1: computing global std_dev ---")
    all_deltas = []
    for cat, sketch in all_splits["train"]:
        seq = convert_raw_sketch_to_delta_sequence(sketch)
        if seq.size:
            all_deltas.extend(seq[:,0].tolist())
            all_deltas.extend(seq[:,1].tolist())

    if not all_deltas:
        std_dev = 1.0
        print("No deltas found; defaulting std_dev=1.0")
    else:
        std_dev = float(np.std(all_deltas))
        if std_dev < 1e-6:
            std_dev = 1.0
        print(f"Computed std_dev = {std_dev:.4f}")

    # Pass 2: normalize, save by split/category
    print("\n--- Pass 2: normalizing & saving ---")
    for split, items in all_splits.items():
        by_cat = {}
        for cat, sketch in items:
            seq = convert_raw_sketch_to_delta_sequence(sketch)
            if seq.size == 0:
                continue
            seq[:,:2] /= std_dev
            by_cat.setdefault(cat, []).append(seq)

        for cat, seqs in by_cat.items():
            out_path = os.path.join(proc_base, split, f"{cat}.npy")
            np.save(out_path, np.array(seqs, dtype=object))
            print(f" Saved {len(seqs)} sketches for '{cat}' → {split}")

    # Save config
    cfg = {
        "quickdraw_std_dev": std_dev,
        "category_map": category_map,
        "categories": categories,
        "train_ratio": train_r,
        "val_ratio": val_r,
        "note": mode: dx,dy,p-states normalized, split by category"
    }
    cfg_path = os.path.join(proc_base, CONFIG_FILENAME)
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"\nPreprocessing complete. Config written to {cfg_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, convert, normalize, and split QuickDraw (vector) dataset."
    )
    parser.add_argument(
        "--categories", nargs="+", default=DEFAULT_CATEGORIES,
        help="List of QuickDraw categories (simplified) to process."
    )
    parser.add_argument(
        "--raw_dir", default=RAW_DOWNLOAD_DIR_ROOT,
        help="Directory for raw .ndjson files."
    )
    parser.add_argument(
        "--processed_dir", default=PROCESSED_DATA_DIR_ROOT,
        help="Directory to save processed data."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=TRAIN_RATIO,
        help="Fraction of sketches for training."
    )
    parser.add_argument(
        "--val_ratio", type=float, default=VAL_RATIO,
        help="Fraction of sketches for validation."
    )
    args = parser.parse_args()
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    main(
        args.categories,
        args.raw_dir,
        args.processed_dir,
        args.train_ratio,
        args.val_ratio
    )
