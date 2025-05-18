#!/usr/bin/env python3
# data/preprocess_dataset_quickdraw.py

import os
import json
import random
import argparse
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

# --- Configuration ---
BASE_DOWNLOAD_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
CATEGORIES_LIST_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
DEFAULT_CATEGORIES = [
    "cat", "dog", "apple", "car", "tree", "house", "bicycle", "bird", "face", "airplane"
]
FALLBACK_CATEGORIES = DEFAULT_CATEGORIES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DOWNLOAD_DIR_ROOT = os.path.join(SCRIPT_DIR, "..", "downloaded_data")
PROCESSED_DATA_DIR_ROOT = os.path.join(SCRIPT_DIR, "..", "processed_data")

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
RASTER_IMG_SIZE = 224
CONFIG_FILENAME = "quickdraw_config.json"

# --- Helper Functions (unchanged) ---
def download_all_categories_list(url=CATEGORIES_LIST_URL):
    print(f"Attempting to download category list from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        categories = [line.strip() for line in response.text.split('\n') if line.strip()]
        if categories:
            print(f"Successfully downloaded {len(categories)} category names.")
            return sorted(categories)
        else:
            print("Downloaded category list is empty."); return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading category list: {e}"); return None

def download_category_file(category_name, target_dir, skip_actual_download=False):
    url_category_name = category_name.replace(" ", "%20")
    raw_cat_dir = os.path.join(target_dir, "quickdraw_raw")
    os.makedirs(raw_cat_dir, exist_ok=True)
    file_name = f"{category_name}.ndjson"
    url = f"{BASE_DOWNLOAD_URL}{url_category_name}.ndjson"
    target_path = os.path.join(raw_cat_dir, file_name)

    if os.path.exists(target_path):
        return target_path, True

    if skip_actual_download:
        print(f"File '{file_name}' not found locally and --skip_raw_download is set. Skipping download.")
        return target_path, False

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f, tqdm(
            desc=f"Downloading {category_name[:20]:20}",
            total=int(response.headers.get('content-length', 0)),
            unit='iB', unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return target_path, True
    except Exception:
        if os.path.exists(target_path):
            os.remove(target_path)
        return target_path, False

def load_simplified_drawings(ndjson_path, max_items_per_category=None):
    drawings = []
    try:
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_items_per_category is not None and i >= max_items_per_category:
                    break
                try:
                    obj = json.loads(line)
                    if obj.get('recognized', False):
                        drawings.append(obj['drawing'])
                except:
                    continue
    except FileNotFoundError:
        print(f"Warning: File not found during load: {ndjson_path}. Skipping.")
    except Exception as e:
        print(f"Error reading {ndjson_path}: {e}")
    return drawings

def convert_raw_sketch_to_delta_sequence(raw_sketch_strokes):
    points = []; last_x, last_y = 0, 0
    num_strokes = len(raw_sketch_strokes)
    if num_strokes == 0:
        return np.zeros((0,5), dtype=np.float32)
    for stroke_idx, stroke in enumerate(raw_sketch_strokes):
        xs, ys = stroke
        if not xs or len(xs) != len(ys):
            continue
        dx0 = xs[0] - last_x; dy0 = ys[0] - last_y
        if stroke_idx > 0:
            points.append([dx0, dy0, 0, 1, 0])
            dx0, dy0 = 0, 0
        is_last = (stroke_idx == num_strokes - 1) and (len(xs) == 1)
        p_state = [0,0,1] if is_last else [1,0,0]
        points.append([dx0, dy0] + p_state)
        last_x, last_y = xs[0], ys[0]
        for i in range(1, len(xs)):
            dx = xs[i] - last_x; dy = ys[i] - last_y
            last_x, last_y = xs[i], ys[i]
            is_last = (stroke_idx == num_strokes - 1) and (i == len(xs) - 1)
            p_state = [0,0,1] if is_last else [1,0,0]
            points.append([dx, dy] + p_state)
        if stroke_idx < num_strokes - 1 and points and points[-1][4] == 0:
            points[-1][2], points[-1][3] = 0, 1
    if points:
        points[-1][2:] = [0, 0, 1]
    return np.array(points, dtype=np.float32)

def rasterize_sequence_to_pil_image(seq, image_size, line_thickness=2, padding=0.02):
    image = Image.new("L", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    abs_segs = []; x0 = y0 = 0.0
    for dx, dy, p_down, p_up, p_eos in seq:
        x1, y1 = x0 + dx, y0 + dy
        if p_down > 0.5 and (x0, y0) != (0,0):
            abs_segs.append(((x0,y0),(x1,y1)))
        x0, y0 = x1, y1
        if p_eos > 0.5:
            break
    if not abs_segs:
        return image.convert("1")
    xs = [p[0][0] for p in abs_segs] + [p[1][0] for p in abs_segs]
    ys = [p[0][1] for p in abs_segs] + [p[1][1] for p in abs_segs]
    if not xs or not ys:
        return image.convert("1")
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = maxx-minx, maxy-miny
    if w < 1e-6 and h < 1e-6:
        mx, my = abs_segs[0][0]
        cx = image_size/2 + (mx-minx)
        cy = image_size/2 + (my-miny)
        draw.ellipse([(cx-1,cy-1),(cx+1,cy+1)], fill="black")
        return image.convert("1")
    canvas_w = image_size*(1-2*padding)
    canvas_h = image_size*(1-2*padding)
    scale = min(canvas_w/w if w>0 else 1, canvas_h/h if h>0 else 1)
    offx = (image_size - w*scale)/2
    offy = (image_size - h*scale)/2
    for (x0,y0),(x1,y1) in abs_segs:
        sx0 = (x0-minx)*scale + offx
        sy0 = (y0-miny)*scale + offy
        sx1 = (x1-minx)*scale + offx
        sy1 = (y1-miny)*scale + offy
        draw.line([(sx0,sy0),(sx1,sy1)], fill="black", width=line_thickness)
    return image.convert("1")

def normalize_and_rasterize(args):
    seq_array, std_dev, img_size = args
    try:
        norm = seq_array.copy()
        norm[:, :2] /= std_dev
        pil = rasterize_sequence_to_pil_image(norm, img_size)
        return norm, pil
    except Exception:
        return None, None

# --- Main (parallelized) ---
def main(categories_initial, raw_dir, proc_dir, train_r, val_r, max_items, skip_raw_download):
    print(f"--- Starting QuickDraw Preprocessing ({'skip_download' if skip_raw_download else 'full_download'}) ---")
    if skip_raw_download:
        print("!!! --skip_raw_download is active: only processing existing .ndjson files. !!!")
    if max_items:
        print(f"Max items per category: {max_items}")

    # Prepare dirs
    vector_dir = os.path.join(proc_dir, "quickdraw_vector")
    raster_dir = os.path.join(proc_dir, "quickdraw_raster")
    raw_ndjson_dir = os.path.join(raw_dir, "quickdraw_raw")
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(raster_dir, exist_ok=True)
    for split in ("train","val","test"):
        os.makedirs(os.path.join(vector_dir, split), exist_ok=True)
        os.makedirs(os.path.join(raster_dir, split), exist_ok=True)

    # Determine which categories to process
    if skip_raw_download:
        existing = set(f for f in os.listdir(raw_ndjson_dir) if f.endswith(".ndjson")) if os.path.isdir(raw_ndjson_dir) else set()
        cats = [c for c in categories_initial if f"{c}.ndjson" in existing]
        if not cats:
            print(f"No local .ndjson found in {raw_ndjson_dir} for requested categories. Exiting.")
            return
        categories = cats
    else:
        categories = categories_initial

    if not categories:
        print("No categories to process. Exiting."); return

    print(f"Final list of categories ({len(categories)}): {categories}")

    # Pre-create per-category raster dirs
    for split in ("train","val","test"):
        for c in categories:
            os.makedirs(os.path.join(raster_dir, split, c), exist_ok=True)

    # Build category map
    category_map = {c:i for i,c in enumerate(categories)}

    # --- Parallel download & load ---
    all_sketches = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_category_file, c, raw_dir, skip_raw_download): c
            for c in categories
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            cat = futures[fut]
            try:
                path, ok = fut.result()
                if not ok:
                    print(f"  -> Skipping '{cat}' (download/file error).")
                    continue
                drawings = load_simplified_drawings(path, max_items_per_category=max_items)
                if drawings:
                    all_sketches[cat] = drawings
                else:
                    print(f"  -> No valid drawings for '{cat}'.")
            except Exception as e:
                print(f"  -> Unexpected error for '{cat}': {e}")

    if not all_sketches:
        print("No sketches loaded. Aborting."); return

    # --- Pass 1: convert to delta sequences in parallel ---
    num_workers = max(1, cpu_count()//2)
    all_seq = {}
    train_deltas = []
    for cat, drawings in all_sketches.items():
        with Pool(num_workers) as pool:
            seqs = list(tqdm(
                pool.imap(convert_raw_sketch_to_delta_sequence, drawings),
                total=len(drawings),
                desc=f"Δ→seq ({cat})"
            ))
        seqs = [s for s in seqs if s is not None and s.size>0]
        all_seq[cat] = seqs
        # collect for std dev
        n_train = int(train_r * len(seqs))
        for seq in seqs[:n_train]:
            train_deltas.extend(seq[:,0])
            train_deltas.extend(seq[:,1])

    std_dev = float(np.std(train_deltas)) if train_deltas else 1.0
    std_dev = max(std_dev, 1e-6)
    print(f"Computed global std_dev = {std_dev:.6f}")

    # --- Pass 2: normalize + rasterize in parallel, then split & save ---
    for cat, seqs in all_seq.items():
        args = [(seq, std_dev, RASTER_IMG_SIZE) for seq in seqs]
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(normalize_and_rasterize, args),
                total=len(args),
                desc=f"Norm+Raster ({cat})"
            ))
        valid = [(n, img) for n, img in results if img is not None]
        if not valid:
            print(f"  -> No valid raster results for '{cat}'. Skipping.")
            continue

        vecs, imgs = zip(*valid)
        combined = list(zip(vecs, imgs))
        random.shuffle(combined)
        vecs, imgs = zip(*combined)
        total = len(vecs)
        n_train = int(train_r * total)
        n_val   = int(val_r * total)
        splits = {
            "train": (vecs[:n_train], imgs[:n_train]),
            "val":   (vecs[n_train:n_train+n_val], imgs[n_train:n_train+n_val]),
            "test":  (vecs[n_train+n_val:], imgs[n_train+n_val:])
        }
        for split, (vseqs, rimgs) in splits.items():
            if not vseqs:
                continue
            # save vectors
            vec_path = os.path.join(vector_dir, split, f"{cat}.npy")
            np.save(vec_path, np.array(vseqs, dtype=object))
            # save rasters
            out_dir = os.path.join(raster_dir, split, cat)
            for i, im in enumerate(rimgs):
                im.save(os.path.join(out_dir, f"sketch_{i:05d}.png"))

    # --- Write config ---
    cfg = {
        "dataset_name": "quickdraw",
        "quickdraw_std_dev": std_dev,
        "category_map": category_map,
        "categories_processed": categories,
        "train_ratio": train_r,
        "val_ratio": val_r,
        "test_ratio": round(1-train_r-val_r, 2),
        "max_items_per_category_processed": max_items or "all",
        "vector_data_path_relative": "quickdraw_vector",
        "raster_data_path_relative": "quickdraw_raster",
        "raster_image_size": RASTER_IMG_SIZE,
        "data_format_note_vector": "Each .npy: list of [N×5] arrays (dx,dy,p_down,p_up,p_eos), normalized dx/dy.",
        "data_format_note_raster": f"PNG images in quickdraw_raster, {RASTER_IMG_SIZE}×{RASTER_IMG_SIZE}, binary."
    }
    cfg_path = os.path.join(vector_dir, CONFIG_FILENAME)
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=4)

    print(f"\nPreprocessing complete! Data in:\n  Vectors: {vector_dir}\n  Rasters: {raster_dir}\nConfig: {cfg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuickDraw Preprocessing (Vector & Raster).")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--all_categories", action="store_true")
    parser.add_argument("--skip_raw_download", action="store_true",
                        help="Only process existing .ndjson files; do not fetch new ones.")
    parser.add_argument("--raw_dir", default=RAW_DOWNLOAD_DIR_ROOT)
    parser.add_argument("--processed_dir", default=PROCESSED_DATA_DIR_ROOT)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--max_items_per_category", type=int, default=None)

    args = parser.parse_args()
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    if args.all_categories:
        cats = download_all_categories_list() or FALLBACK_CATEGORIES
        print(f"Targeting {len(cats)} categories.")
    elif args.categories:
        cats = sorted(set(args.categories))
        print(f"Targeting specified categories: {cats}")
    else:
        cats = sorted(DEFAULT_CATEGORIES)
        print(f"Targeting default categories: {cats}")

    print(f"Raw dir:       {os.path.abspath(args.raw_dir)}")
    print(f"Processed dir: {os.path.abspath(args.processed_dir)}\n")

    main(
        categories_initial=cats,
        raw_dir=args.raw_dir,
        proc_dir=args.processed_dir,
        train_r=args.train_ratio,
        val_r=args.val_ratio,
        max_items=args.max_items_per_category,
        skip_raw_download=args.skip_raw_download
    )
