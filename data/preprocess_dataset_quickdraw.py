#!/usr/bin/env python3

import os
import json
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, Manager # Manager for shared progress dict if needed
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw
import traceback # For more detailed error logging

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

BASE_DOWNLOAD_URL    = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
CATEGORIES_LIST_URL  = (
    "https://raw.githubusercontent.com/"
    "googlecreativelab/quickdraw-dataset/master/categories.txt"
)
DEFAULT_CATEGORIES   = ["cat","dog","apple","car","tree","house",
                        "bicycle","bird","face","airplane"]
FALLBACK_CATEGORIES  = DEFAULT_CATEGORIES 
TRAIN_RATIO          = 0.80
VAL_RATIO            = 0.10
RASTER_IMG_SIZE      = 224 
CONFIG_FILENAME      = "quickdraw_config.json" 

SCRIPT_DIR           = os.path.dirname(os.path.abspath(__file__))
RAW_DOWNLOAD_DIR     = os.path.join(SCRIPT_DIR, "..", "downloaded_data", "quickdraw_raw")
PROCESSED_DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "processed_data")

# ─── HELPER FUNCTIONS (Including the missing one) ───────────────────────────

def download_all_categories_list(url=CATEGORIES_LIST_URL):
    """Downloads the list of all categories from the given URL."""
    print(f"Attempting to download category list from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        categories = [line.strip() for line in response.text.split('\n') if line.strip()]
        if categories:
            print(f"Successfully downloaded {len(categories)} category names.")
            return sorted(categories) # Sort for consistency
        else:
            print("Downloaded category list is empty.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading category list: {e}")
        return None

_thread_slots = {} 
_slot_lock    = threading.Lock() 

def _get_thread_slot(max_slots: int) -> int:
    tid = threading.get_ident() 
    with _slot_lock: 
        if tid not in _thread_slots:
            _thread_slots[tid] = len(_thread_slots) % max_slots
        return _thread_slots[tid]

def download_one(cat_name: str, target_dir: str, max_slots: int):
    os.makedirs(target_dir, exist_ok=True) 
    url_safe_cat_name = cat_name.replace(" ", "%20") 
    output_path       = os.path.join(target_dir, f"{cat_name}.ndjson")
    download_url      = BASE_DOWNLOAD_URL + url_safe_cat_name + ".ndjson"

    if os.path.exists(output_path):
        try:
            head_response = requests.head(download_url, timeout=5) 
            head_response.raise_for_status() 
            expected_total_size = int(head_response.headers.get("Content-Length", 0))
            if expected_total_size and os.path.getsize(output_path) == expected_total_size:
                return output_path, True 
        except requests.exceptions.RequestException:
            pass 
        if os.path.exists(output_path): 
            try: os.remove(output_path)
            except OSError: pass
    try:
        response = requests.get(download_url, stream=True, timeout=30) 
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0))
        tqdm_position = 1 + _get_thread_slot(max_slots) 
        with open(output_path, "wb") as f, \
             tqdm(total=total_size, unit="B", unit_scale=True,
                  desc=f"{cat_name[:15]}...", 
                  position=tqdm_position, 
                  leave=False, 
                  mininterval=0.5 
                  ) as pbar:
            for chunk in response.iter_content(chunk_size=64_000): 
                if chunk: 
                    f.write(chunk)
                    pbar.update(len(chunk))
        return output_path, True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {cat_name}: {e}")
        if os.path.exists(output_path): 
            try: os.remove(output_path)
            except OSError: pass
        return output_path, False
    except Exception as e_gen: 
        print(f"Generic error for {cat_name} during download: {e_gen}")
        return output_path, False

def iter_drawings(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue 
            if obj.get("recognized", False) and "drawing" in obj: 
                yield obj["drawing"]

def convert_raw_sketch_to_delta_sequence(strokes):
    points = []
    last_x = last_y = 0
    for stroke_index, stroke in enumerate(strokes):
        x_coords, y_coords = stroke
        if not x_coords or len(x_coords) != len(y_coords): continue 
        delta_x0 = x_coords[0] - last_x; delta_y0 = y_coords[0] - last_y
        if stroke_index > 0: 
            points.append([delta_x0, delta_y0, 0, 1, 0]) 
            delta_x0, delta_y0 = 0, 0 
        is_last_point_of_sketch = (stroke_index == len(strokes) - 1 and len(x_coords) == 1)
        pen_state = [0, 0, 1] if is_last_point_of_sketch else [1, 0, 0] 
        points.append([delta_x0, delta_y0] + pen_state)
        last_x, last_y = x_coords[0], y_coords[0]
        for i in range(1, len(x_coords)):
            delta_x = x_coords[i] - last_x; delta_y = y_coords[i] - last_y
            last_x, last_y = x_coords[i], y_coords[i]
            is_last_point_of_sketch = (stroke_index == len(strokes) - 1 and i == len(x_coords) - 1)
            pen_state = [0, 0, 1] if is_last_point_of_sketch else [1, 0, 0] 
            points.append([delta_x, delta_y] + pen_state)
        if stroke_index < len(strokes) - 1:
            if points and points[-1][4] == 0: 
                points[-1][2] = 0; points[-1][3] = 1 
    if points:
        points[-1][2] = 0; points[-1][3] = 0; points[-1][4] = 1 
    return np.array(points, dtype=np.float32)

def rasterize_sequence_to_pil_image(normalized_delta_sequence, image_size: int, 
                                    line_thickness=2, padding_percent=0.02):
    pil_image = Image.new("L", (image_size, image_size), "white") 
    draw_context = ImageDraw.Draw(pil_image)
    absolute_segments = [] 
    current_abs_x, current_abs_y = 0.0, 0.0 
    for i in range(normalized_delta_sequence.shape[0]):
        dx, dy, p_down, p_up, p_eos = normalized_delta_sequence[i]
        next_abs_x = current_abs_x + dx; next_abs_y = current_abs_y + dy
        if p_down > 0.5: 
            absolute_segments.append(((current_abs_x, current_abs_y), (next_abs_x, next_abs_y)))
        current_abs_x, current_abs_y = next_abs_x, next_abs_y
        if p_eos > 0.5: break 
    if not absolute_segments: return pil_image.convert("1")
    all_x = [c for seg in absolute_segments for c in (seg[0][0], seg[1][0])]
    all_y = [c for seg in absolute_segments for c in (seg[0][1], seg[1][1])]
    if not all_x or not all_y: return pil_image.convert("1")
    min_coord_x, max_coord_x = min(all_x), max(all_x)
    min_coord_y, max_coord_y = min(all_y), max(all_y)
    sketch_pixel_width = max_coord_x - min_coord_x
    sketch_pixel_height = max_coord_y - min_coord_y
    if sketch_pixel_width < 1e-6 and sketch_pixel_height < 1e-6: 
        if absolute_segments:
            pt_x_norm = absolute_segments[0][0][0]; pt_y_norm = absolute_segments[0][0][1]
            draw_x_canvas = image_size / 2 + (pt_x_norm - min_coord_x) 
            draw_y_canvas = image_size / 2 + (pt_y_norm - min_coord_y)
            draw_context.ellipse([(draw_x_canvas-1, draw_y_canvas-1), (draw_x_canvas+1, draw_y_canvas+1)], fill="black")
        return pil_image.convert("1")
    target_draw_width = image_size * (1 - 2 * padding_percent)
    target_draw_height = image_size * (1 - 2 * padding_percent)
    scale = 1.0
    if sketch_pixel_width > 1e-6: scale = target_draw_width / sketch_pixel_width
    if sketch_pixel_height > 1e-6: scale = min(scale, target_draw_height / sketch_pixel_height)
    offset_x_on_canvas = (image_size - (sketch_pixel_width * scale)) / 2.0
    offset_y_on_canvas = (image_size - (sketch_pixel_height * scale)) / 2.0
    for (x0_abs, y0_abs), (x1_abs, y1_abs) in absolute_segments:
        sx0 = ((x0_abs - min_coord_x) * scale) + offset_x_on_canvas
        sy0 = ((y0_abs - min_coord_y) * scale) + offset_y_on_canvas
        sx1 = ((x1_abs - min_coord_x) * scale) + offset_x_on_canvas
        sy1 = ((y1_abs - min_coord_y) * scale) + offset_y_on_canvas
        draw_context.line([(sx0, sy0), (sx1, sy1)], width=line_thickness, fill="black")
    del draw_context
    return pil_image.convert("1") 

def normalize_sequence_task(args_tuple):
    raw_strokes, global_std_dev = args_tuple
    try:
        delta_sequence = convert_raw_sketch_to_delta_sequence(raw_strokes)
        if delta_sequence.size == 0: return None
        normalized_sequence = delta_sequence.copy()
        normalized_sequence[:, :2] /= global_std_dev
        return normalized_sequence
    except Exception as e:
        return None

def _rasterize_save_task(args_tuple): 
    idx, normalized_sequence, raster_split_cat_dir, img_size = args_tuple
    try:
        pil_img = rasterize_sequence_to_pil_image(normalized_sequence, img_size)
        img_filename = f"sketch_{idx:05d}.png"
        output_path = os.path.join(raster_split_cat_dir, img_filename)
        pil_img.save(output_path)
        return True 
    except Exception as e:
        return False

def run_pass1_for_category(cat_name, raw_dir, vec_base, std_dev, train_r, val_r, n_workers, max_items_cat=None):
    path = os.path.join(raw_dir, f"{cat_name}.ndjson")
    if not os.path.exists(path):
        print(f"NDJSON file not found for {cat_name} at {path}. Skipping Pass 1.")
        return
        
    tasks = [(strokes, std_dev) for strokes in iter_drawings(path)]
    if max_items_cat is not None:
        tasks = tasks[:max_items_cat]

    if not tasks:
        print(f"No drawings to process for {cat_name} in Pass 1.")
        return

    normalized_sequences = []
    pool_processes = n_workers if n_workers and n_workers > 0 else 1
    
    if pool_processes == 1:
        for task_args in tasks: 
            seq = normalize_sequence_task(task_args)
            if seq is not None: normalized_sequences.append(seq)
    else:
        with Pool(processes=pool_processes) as p:
            results = list(p.imap(normalize_sequence_task, tasks, chunksize=max(1, len(tasks)//(pool_processes*4)))) 
            normalized_sequences = [seq for seq in results if seq is not None]

    if not normalized_sequences:
        print(f"No sequences normalized for {cat_name}. Skipping save.")
        return
        
    random.shuffle(normalized_sequences)
    total = len(normalized_sequences)
    ntr   = int(train_r * total)
    nvl   = int(val_r   * total)
    splits = {
        "train": normalized_sequences[:ntr],
        "val":   normalized_sequences[ntr:ntr+nvl],
        "test":  normalized_sequences[ntr+nvl:]
    }
    for sp, seqs_in_split in splits.items():
        if not seqs_in_split: continue
        out_path = os.path.join(vec_base, sp, f"{cat_name}.npy")
        np.save(out_path, np.array(seqs_in_split, dtype=object))

def run_pass2_for_category(cat_name, vec_base, ras_base, img_size, n_workers):
    for sp in ("train","val","test"):
        vec_path = os.path.join(vec_base, sp, f"{cat_name}.npy")
        out_dir  = os.path.join(ras_base, sp, cat_name) 
        
        if not os.path.exists(vec_path):
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        sequences_to_rasterize = np.load(vec_path, allow_pickle=True)
        
        if sequences_to_rasterize.size == 0:
            continue

        existing_indices = set()
        try:
            existing_indices = {
                int(fn.split("_")[1].split(".")[0])
                for fn in os.listdir(out_dir)
                if fn.startswith("sketch_") and fn.endswith(".png")
            }
        except FileNotFoundError: 
            pass
            
        tasks = [
            (idx, seq, out_dir, img_size) 
            for idx, seq in enumerate(sequences_to_rasterize)
            if idx not in existing_indices 
        ]

        if not tasks:
            continue
        
        pool_processes = n_workers if n_workers and n_workers > 0 else 1
        if pool_processes == 1:
            for task_args in tasks: 
                _rasterize_save_task(task_args)
        else:
            with Pool(processes=pool_processes) as p:
                list(p.imap_unordered(_rasterize_save_task, tasks, chunksize=max(1, len(tasks)//(pool_processes*4)))) 
    
def process_category_task(args_bundle):
    cat_name, raw_dir, vec_base, ras_base, std_dev, train_r, val_r, n_workers, img_size, max_items_cat, progress_dict_cat = args_bundle
    current_progress = progress_dict_cat.get(cat_name, {})
    try:
        if not current_progress.get("pass1_done", False):
            run_pass1_for_category(cat_name, raw_dir, vec_base, std_dev, train_r, val_r, n_workers, max_items_cat)
            current_progress["pass1_done"] = True
        if not current_progress.get("pass2_done", False):
            run_pass2_for_category(cat_name, vec_base, ras_base, img_size, n_workers)
            current_progress["pass2_done"] = True
        return cat_name, current_progress, None 
    except Exception as e:
        print(f"!!! Error processing category {cat_name} in worker {os.getpid()}: {e}")
        return cat_name, current_progress, str(e) 

_cat_slots_stats = {} 
def init_stats_worker(cat_slots_arg): global _cat_slots_stats; _cat_slots_stats = cat_slots_arg
def stats_for_category_task(args_tuple):
    category_name, raw_ndjson_file_path, max_items = args_tuple
    n_category, mean_category, M2_category = 0.0, 0.0, 0.0
    count = 0
    for raw_strokes in iter_drawings(raw_ndjson_file_path):
        if max_items is not None and count >= max_items: break
        delta_sequence = convert_raw_sketch_to_delta_sequence(raw_strokes)
        if delta_sequence.size > 0:
            deltas_for_std = np.concatenate((delta_sequence[:, 0], delta_sequence[:, 1]))
            for x_val in deltas_for_std:
                n_category += 1; delta = x_val - mean_category; mean_category += delta / n_category
                M2_category += delta * (x_val - mean_category) 
        count +=1
    return category_name, n_category, mean_category, M2_category

def merge_stats_parallel(stat_results_list):
    total_n, total_mean, total_M2 = 0.0, 0.0, 0.0
    # Corrected: stat_results_list contains tuples of (n_cat, mean_cat, M2_cat)
    # The cat_name was used to populate stats_progress, but not directly passed in this list structure.
    for n_cat, mean_cat, M2_cat in stat_results_list: 
        if n_cat == 0: continue 
        if total_n == 0: 
            total_n, total_mean, total_M2 = n_cat, mean_cat, M2_cat
        else:
            delta = mean_cat - total_mean; new_total_n = total_n + n_cat
            total_mean += delta * (n_cat / new_total_n)
            total_M2 += M2_cat + (delta**2) * (total_n * n_cat / new_total_n)
            total_n = new_total_n
    return total_n, total_mean, total_M2

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuickDraw Preproc (incremental)")
    parser.add_argument("--step", choices=["download","stats","process","all"], default="all")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--all_categories", action="store_true")
    parser.add_argument("--skip_raw_download", action="store_true")
    parser.add_argument("--raw_dir", default=RAW_DOWNLOAD_DIR)
    parser.add_argument("--processed_dir", default=PROCESSED_DATA_DIR)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() if os.cpu_count() else 1, 8))
    parser.add_argument("--max_items_per_category", type=int, default=None)
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0: raise ValueError("Train + Val ratio must be < 1.0.")
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    print(f"Raw Dir: {os.path.abspath(args.raw_dir)}")
    print(f"Processed Dir Base: {os.path.abspath(args.processed_dir)}")

    target_categories = []
    if args.all_categories:
        downloaded_cats = download_all_categories_list() 
        target_categories = downloaded_cats if downloaded_cats else FALLBACK_CATEGORIES
    elif args.categories: target_categories = sorted(list(set(args.categories)))
    else: target_categories = sorted(DEFAULT_CATEGORIES)
    print(f"Targeting {len(target_categories)} categories.")
    if not target_categories: print("No categories. Exiting."); return

    vector_output_base = os.path.join(args.processed_dir, "quickdraw_vector")
    raster_output_base = os.path.join(args.processed_dir, "quickdraw_raster")
    stats_progress_file = os.path.join(vector_output_base, "stats_progress.json")
    main_config_file = os.path.join(vector_output_base, CONFIG_FILENAME)
    process_stage_progress_file = os.path.join(args.processed_dir, "process_stage_progress.json")

    for sp in ("train","val","test"):
        os.makedirs(os.path.join(vector_output_base, sp), exist_ok=True)
        os.makedirs(os.path.join(raster_output_base, sp), exist_ok=True)
        for cat_name in target_categories: 
            os.makedirs(os.path.join(raster_output_base, sp, cat_name), exist_ok=True)

    if args.step in ("download","all"):
        print("\n=== STAGE: Download Raw NDJSON Files ===")
        if args.skip_raw_download: print("→ Skipping download attempts (--skip_raw_download set).")
        
        categories_with_raw_files = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_cat = {executor.submit(download_one, cat, args.raw_dir, args.threads): cat for cat in target_categories}
            for future in tqdm(as_completed(future_to_cat), total=len(target_categories), desc="Download Progress"):
                cat_name = future_to_cat[future]
                try:
                    _, success = future.result()
                    if success: categories_with_raw_files.append(cat_name)
                    else: print(f"Warning: Failed for '{cat_name}'. Will be skipped.")
                except Exception as exc: print(f"Exception for '{cat_name}' during download: {exc}")
        
        target_categories = sorted(list(set(categories_with_raw_files)))
        if not target_categories: print("No raw files. Exiting."); return
        print(f"Proceeding with {len(target_categories)} categories with raw data.")
        if args.step == "download": return

    global_std_dev = 1.0
    if args.step in ("stats","all"):
        print("\n=== STAGE: Calculate Global Standard Deviation ===")
        stats_progress = {}
        if os.path.exists(stats_progress_file):
            try:
                with open(stats_progress_file, "r") as f: stats_progress = json.load(f)
                print(f"Loaded stats progress for {len(stats_progress)} categories.")
            except json.JSONDecodeError: print(f"Warning: Could not parse {stats_progress_file}.")
        
        categories_for_stats = [cat for cat in target_categories if cat not in stats_progress]
        if not categories_for_stats:
            print("→ Stats already computed for all target categories.")
            if os.path.exists(main_config_file):
                with open(main_config_file, "r") as f: cfg = json.load(f)
                global_std_dev = cfg.get("quickdraw_std_dev", 1.0)
            else: print("Warning: Main config not found, std_dev might be default.")
        else:
            print(f"Calculating stats for {len(categories_for_stats)} new/remaining categories...")
            stats_tasks = [(cat, os.path.join(args.raw_dir, f"{cat}.ndjson"), args.max_items_per_category) 
                           for cat in categories_for_stats
                           if os.path.exists(os.path.join(args.raw_dir, f"{cat}.ndjson"))] 
            
            if not stats_tasks:
                print("No valid .ndjson files found for categories needing stats. Skipping stats calculation.")
            else:
                with Pool(processes=args.threads) as pool:
                    results_iterator = pool.imap(stats_for_category_task, stats_tasks)
                    for cat_name, n_cat, mean_cat, M2_cat in tqdm(results_iterator, total=len(stats_tasks), desc="Stats Calculation"):
                        stats_progress[cat_name] = [float(n_cat), float(mean_cat), float(M2_cat)]
                        with open(stats_progress_file, "w") as f: json.dump(stats_progress, f, indent=2)
            
            all_cat_stats_for_merge = []
            for cat_name_stat in target_categories: 
                if cat_name_stat in stats_progress: 
                    all_cat_stats_for_merge.append(stats_progress[cat_name_stat]) # This is a list of [n, mean, M2]
            
            if not all_cat_stats_for_merge: print("No stats to compute global std_dev.")
            else:
                total_n_all, _, total_M2_all = merge_stats_parallel(all_cat_stats_for_merge) # Pass list of lists
                if total_n_all > 1: global_std_dev = float(np.sqrt(total_M2_all / (total_n_all - 1)))
                global_std_dev = max(global_std_dev, 1e-6)
            print(f"→ Global std_dev calculated: {global_std_dev:.6f}")
            current_config = {}
            if os.path.exists(main_config_file):
                with open(main_config_file, "r") as f: current_config = json.load(f)
            current_config.update({
                "dataset_name": "quickdraw", "quickdraw_std_dev": global_std_dev,
                "categories_processed": target_categories, 
                "category_map": {cat: i for i, cat in enumerate(target_categories)},
                "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
                "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 2),
                "max_items_per_category_processed": args.max_items_per_category if args.max_items_per_category else "all",
                "vector_data_path_relative": "quickdraw_vector", "raster_data_path_relative": "quickdraw_raster",
                "raster_image_size": RASTER_IMG_SIZE,
                "data_format_note_vector": "Each .npy file: list of sequences [N,5] (dx,dy,p0,p1,p2), dx,dy normalized.",
                "data_format_note_raster": f"PNG images, {RASTER_IMG_SIZE}x{RASTER_IMG_SIZE}, binary."})
            with open(main_config_file, "w") as f: json.dump(current_config, f, indent=4)
            print(f"  Main config updated at {main_config_file}")
        if args.step == "stats": return

    if not os.path.exists(main_config_file):
        print(f"Error: Main config {main_config_file} missing. Run 'stats' step. Exiting."); return
    with open(main_config_file, "r") as f: cfg = json.load(f)
    global_std_dev = cfg.get("quickdraw_std_dev", 1.0)
    categories_for_processing = cfg.get("categories_processed", target_categories)
    if not categories_for_processing: print("No categories in config. Exiting."); return
    
    if args.step in ("process","all"):
        print("\n=== STAGE: Process Data (Vector & Raster) ===")
        process_progress = {}
        if os.path.exists(process_stage_progress_file):
            try:
                with open(process_stage_progress_file, "r") as f: process_progress = json.load(f)
                print(f"Loaded processing stage progress for {len(process_progress)} categories.")
            except json.JSONDecodeError: print(f"Warning: Could not parse {process_stage_progress_file}.")

        tasks_for_processing_stage = []
        for cat_name in categories_for_processing:
            cat_prog = process_progress.get(cat_name, {})
            if not cat_prog.get("pass1_done", False) or not cat_prog.get("pass2_done", False):
                tasks_for_processing_stage.append(
                    (cat_name, args.raw_dir, vector_output_base, raster_output_base, 
                     global_std_dev, args.train_ratio, args.val_ratio, 
                     args.threads, RASTER_IMG_SIZE, args.max_items_per_category, process_progress) 
                )
        
        if not tasks_for_processing_stage:
            print("→ All target categories already processed (pass1 & pass2 done).")
        else:
            print(f"Processing {len(tasks_for_processing_stage)} categories...")
            with ProcessPoolExecutor(max_workers=args.threads) as executor:
                future_to_cat_process = {
                    executor.submit(process_category_task, task_args): task_args[0] 
                    for task_args in tasks_for_processing_stage
                }
                for future in tqdm(as_completed(future_to_cat_process), total=len(tasks_for_processing_stage), desc="Overall Category Processing"):
                    cat_name_completed = future_to_cat_process[future]
                    try:
                        returned_cat_name, updated_cat_prog, error_msg = future.result()
                        if error_msg:
                            print(f"Error processing {returned_cat_name}: {error_msg}")
                        else:
                            process_progress[returned_cat_name] = updated_cat_prog
                            with open(process_stage_progress_file, "w") as f:
                                json.dump(process_progress, f, indent=2)
                    except Exception as exc:
                        print(f"Exception for category '{cat_name_completed}' during processing stage: {exc}")
        if args.step == "process": return
    print("\nPreprocessing pipeline finished.")

if __name__ == "__main__":
    main()
