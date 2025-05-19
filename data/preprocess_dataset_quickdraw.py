#!/usr/bin/env python3

import os
import json
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count # Ensure Pool is imported
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw
import traceback

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

BASE_DOWNLOAD_URL    = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/"
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
RAW_DOWNLOAD_DIR     = os.path.join(SCRIPT_DIR, "..", "downloaded_data", "quickdraw_sketchrnn")
PROCESSED_DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "processed_data")

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

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
    output_path       = os.path.join(target_dir, f"{cat_name}.npz") 
    download_url      = BASE_DOWNLOAD_URL + url_safe_cat_name + ".npz" 

    if os.path.exists(output_path):
        try:
            head_response = requests.head(download_url, timeout=10) 
            head_response.raise_for_status() 
            expected_total_size = int(head_response.headers.get("Content-Length", 0))
            if expected_total_size and os.path.getsize(output_path) == expected_total_size:
                return output_path, True 
            else:
                if os.path.exists(output_path): os.remove(output_path) 
        except requests.exceptions.RequestException: return output_path, True 
        except Exception as e:
            if os.path.exists(output_path): os.remove(output_path)
            
    try:
        response = requests.get(download_url, stream=True, timeout=60) 
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
        if total_size and os.path.getsize(output_path) != total_size:
            print(f"Error: Downloaded size mismatch for {cat_name}. Expected {total_size}, got {os.path.getsize(output_path)}.")
            if os.path.exists(output_path): os.remove(output_path)
            return output_path, False
        return output_path, True
    except requests.exceptions.RequestException as e:
        if os.path.exists(output_path): 
            try: os.remove(output_path)
            except OSError: pass
        return output_path, False
    except Exception as e_gen: 
        return output_path, False

def convert_stroke3_to_stroke5(stroke3_data):
    num_points = stroke3_data.shape[0]
    if num_points == 0:
        return np.zeros((0, 5), dtype=np.float32)
    stroke5_data = np.zeros((num_points, 5), dtype=np.float32)
    stroke5_data[:, :2] = stroke3_data[:, :2]
    stroke5_data[:, 3] = stroke3_data[:, 2] 
    stroke5_data[:, 2] = 1.0 - stroke5_data[:, 3] 
    stroke5_data[:, 4] = 0.0 
    if num_points > 0:
        stroke5_data[num_points - 1, 4] = 1.0 
        stroke5_data[num_points - 1, 2] = 0.0 
        stroke5_data[num_points - 1, 3] = 1.0 
    return stroke5_data.astype(np.float32)

def load_sketches_from_npz(npz_path: str, max_items_per_category=None):
    all_sketches_5_element = []
    try:
        data = np.load(npz_path, encoding='latin1', allow_pickle=True)
        for key in ['train', 'valid', 'test']: 
            if key in data:
                sketches_in_split = data[key]
                for sketch_raw in sketches_in_split:
                    if sketch_raw.ndim == 2 and sketch_raw.shape[1] == 3: 
                        sketch_5elem = convert_stroke3_to_stroke5(sketch_raw)
                        if sketch_5elem.size > 0:
                             all_sketches_5_element.append(sketch_5elem)
                    elif sketch_raw.ndim == 2 and sketch_raw.shape[1] == 5: 
                         all_sketches_5_element.append(sketch_raw.astype(np.float32))
                    if max_items_per_category is not None and len(all_sketches_5_element) >= max_items_per_category:
                        return all_sketches_5_element
        return all_sketches_5_element
    except Exception as e:
        print(f"Error loading or processing NPZ file {npz_path}: {e}")
        return []

def rasterize_sequence_to_pil_image(normalized_delta_sequence, image_size: int, 
                                    line_thickness=2, padding_percent=0.02):
    pil_image = Image.new("L", (image_size, image_size), "white") 
    draw_context = ImageDraw.Draw(pil_image)
    absolute_segments = [] 
    current_abs_x, current_abs_y = 0.0, 0.0 
    for i in range(normalized_delta_sequence.shape[0]):
        dx, dy, p0_draw, p1_lift, p2_eos = normalized_delta_sequence[i] 
        next_abs_x = current_abs_x + dx; next_abs_y = current_abs_y + dy
        if p0_draw > 0.5: 
            absolute_segments.append(((current_abs_x, current_abs_y), (next_abs_x, next_abs_y)))
        current_abs_x, current_abs_y = next_abs_x, next_abs_y
        if p2_eos > 0.5: break 
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
    sketch_5element, global_std_dev = args_tuple
    try:
        if sketch_5element is None or sketch_5element.size == 0 or sketch_5element.shape[1] != 5: 
            return None
        normalized_sequence = sketch_5element.copy()
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

def run_pass1_for_category(cat_name, raw_sketches_5_element, vec_base, std_dev, train_r, val_r, n_workers_ignored): 
    if not raw_sketches_5_element:
        return
    normalized_sequences = []
    for seq_5_elem in raw_sketches_5_element: # Sequential processing
        norm_seq = normalize_sequence_task((seq_5_elem, std_dev))
        if norm_seq is not None:
            normalized_sequences.append(norm_seq)
    if not normalized_sequences:
        return
    random.shuffle(normalized_sequences)
    total = len(normalized_sequences)
    ntr   = int(train_r * total); nvl   = int(val_r   * total)
    splits = {"train": normalized_sequences[:ntr], 
              "val":   normalized_sequences[ntr:ntr+nvl], 
              "test":  normalized_sequences[ntr+nvl:]}
    for sp, seqs_in_split in splits.items():
        if not seqs_in_split: continue
        out_path = os.path.join(vec_base, sp, f"{cat_name}.npy")
        np.save(out_path, np.array(seqs_in_split, dtype=object))

def run_pass2_for_category(cat_name, vec_base, ras_base, img_size, n_workers_ignored):
    for sp in ("train","val","test"):
        vec_path = os.path.join(vec_base, sp, f"{cat_name}.npy")
        out_dir  = os.path.join(ras_base, sp, cat_name) 
        if not os.path.exists(vec_path): continue
        os.makedirs(out_dir, exist_ok=True)
        sequences_to_rasterize = np.load(vec_path, allow_pickle=True)
        if sequences_to_rasterize.size == 0: continue
        existing_indices = set()
        try:
            existing_indices = {int(fn.split("_")[1].split(".")[0]) for fn in os.listdir(out_dir) if fn.startswith("sketch_") and fn.endswith(".png")}
        except FileNotFoundError: pass
        # Sequential processing
        for idx, seq in enumerate(sequences_to_rasterize):
            if idx not in existing_indices:
                _rasterize_save_task((idx, seq, out_dir, img_size))
    
def process_category_task(args_bundle):
    cat_name, raw_sketches_5_element_for_cat, vec_base, ras_base, std_dev, train_r, val_r, _, img_size, progress_dict_cat = args_bundle
    current_progress = progress_dict_cat.get(cat_name, {})
    try:
        if not current_progress.get("pass1_done", False):
            run_pass1_for_category(cat_name, raw_sketches_5_element_for_cat, vec_base, std_dev, train_r, val_r, 1) # n_workers set to 1 (ignored)
            current_progress["pass1_done"] = True
        if not current_progress.get("pass2_done", False): 
            run_pass2_for_category(cat_name, vec_base, ras_base, img_size, 1) # n_workers set to 1 (ignored)
            current_progress["pass2_done"] = True
        return cat_name, current_progress, None 
    except Exception as e:
        print(f"!!! Error processing category {cat_name} in worker {os.getpid()}: {e}")
        return cat_name, current_progress, str(e) 

_cat_slots_stats = {} 
def init_stats_worker(cat_slots_arg): global _cat_slots_stats; _cat_slots_stats = cat_slots_arg
def stats_for_category_task(args_tuple):
    category_name, sketch_sequences_from_npz, max_items = args_tuple 
    n_category, mean_category, M2_category = 0.0, 0.0, 0.0
    sketches_to_process = sketch_sequences_from_npz 
    if max_items is not None and len(sketches_to_process) > max_items:
        sketches_to_process = sketches_to_process[:max_items]
    for delta_sequence in sketches_to_process: 
        if delta_sequence.size > 0 and delta_sequence.ndim == 2 and delta_sequence.shape[1] == 5:
            deltas_for_std = np.concatenate((delta_sequence[:, 0], delta_sequence[:, 1]))
            for x_val in deltas_for_std:
                n_category += 1; delta = x_val - mean_category; mean_category += delta / n_category
                M2_category += delta * (x_val - mean_category) 
    return category_name, n_category, mean_category, M2_category

def merge_stats_parallel(stat_results_list):
    total_n, total_mean, total_M2 = 0.0, 0.0, 0.0
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
    parser = argparse.ArgumentParser(description="QuickDraw Preproc (SketchRNN NPZ, Vector & Raster, Incremental)")
    parser.add_argument("--step", choices=["download","stats","process","all"], default="all") 
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--all_categories", action="store_true")
    parser.add_argument("--skip_raw_download", action="store_true")
    parser.add_argument("--raw_dir", default=RAW_DOWNLOAD_DIR) 
    parser.add_argument("--processed_dir", default=PROCESSED_DATA_DIR)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() if os.cpu_count() else 1, 8),
                        help="Number of threads/processes for parallel tasks (downloads, stats, CATEGORY processing).")
    parser.add_argument("--max_items_per_category", type=int, default=None)
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0: raise ValueError("Train + Val ratio must be < 1.0.")
    os.makedirs(args.raw_dir, exist_ok=True) 
    os.makedirs(args.processed_dir, exist_ok=True)
    print(f"Raw NPZ Dir: {os.path.abspath(args.raw_dir)}")
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
        print("\n=== STAGE: Download Raw NPZ Files ===")
        if args.skip_raw_download: print("→ Skipping download attempts (--skip_raw_download set).")
        categories_with_raw_files = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_cat = {executor.submit(download_one, cat, args.raw_dir, args.threads): cat for cat in target_categories}
            for future in tqdm(as_completed(future_to_cat), total=len(target_categories), desc="Download Progress"):
                cat_name = future_to_cat[future]
                try: _, success = future.result();_ = success and categories_with_raw_files.append(cat_name)
                except Exception as exc: print(f"Download exception for '{cat_name}': {exc}")
        target_categories = sorted(list(set(categories_with_raw_files)))
        if not target_categories: print("No raw files found/downloaded. Exiting."); return
        print(f"Proceeding with {len(target_categories)} categories with raw data.")
        if args.step == "download": return

    all_sketches_by_cat_npz = {} 
    if args.step in ("stats", "process", "all"):
        print("\n--- Loading all sketches from NPZ files (will convert to 5-element) ---")
        for cat_name in tqdm(target_categories, desc="Loading NPZ data"):
            npz_file_path = os.path.join(args.raw_dir, f"{cat_name}.npz")
            if os.path.exists(npz_file_path):
                sketches = load_sketches_from_npz(npz_file_path, args.max_items_per_category)
                if sketches: all_sketches_by_cat_npz[cat_name] = sketches
            else: print(f"Warning: NPZ file for {cat_name} not found. Skipping.")
        if not all_sketches_by_cat_npz: print("No sketches loaded. Exiting."); return
        target_categories = sorted(all_sketches_by_cat_npz.keys()) 
        if not target_categories: print("No categories with loaded sketches. Exiting."); return
        print(f"Successfully loaded and converted sketches for {len(target_categories)} categories.")

    global_std_dev = 1.0
    if args.step in ("stats","all"):
        print("\n=== STAGE: Calculate Global Standard Deviation ===")
        stats_progress = {}
        if os.path.exists(stats_progress_file):
            try:
                with open(stats_progress_file, "r") as f: stats_progress = json.load(f)
                print(f"Loaded stats progress for {len(stats_progress)} categories.")
            except json.JSONDecodeError: print(f"Warning: Could not parse {stats_progress_file}.")
        categories_for_stats = [cat for cat in target_categories if cat not in stats_progress and cat in all_sketches_by_cat_npz]
        if not categories_for_stats:
            print("→ Stats already computed for all target categories with loaded data.")
            if os.path.exists(main_config_file):
                with open(main_config_file, "r") as f: cfg = json.load(f)
                global_std_dev = cfg.get("quickdraw_std_dev", 1.0)
            else: print("Warning: Main config not found, std_dev might be default.")
        else:
            print(f"Calculating stats for {len(categories_for_stats)} new/remaining categories...")
            stats_tasks = [(cat, all_sketches_by_cat_npz[cat], args.max_items_per_category) for cat in categories_for_stats]
            if not stats_tasks: print("No data for categories needing stats.")
            else:
                # Corrected: Now Pool is imported from multiprocessing
                with Pool(processes=args.threads) as pool: 
                    results_iterator = pool.imap(stats_for_category_task, stats_tasks)
                    for cat_name, n_cat, mean_cat, M2_cat in tqdm(results_iterator, total=len(stats_tasks), desc="Stats Calculation"):
                        stats_progress[cat_name] = [float(n_cat), float(mean_cat), float(M2_cat)]
                        with open(stats_progress_file, "w") as f: json.dump(stats_progress, f, indent=2)
            all_cat_stats_for_merge = [stats_progress[cat] for cat in target_categories if cat in stats_progress]
            if not all_cat_stats_for_merge: print("No stats to compute global std_dev.")
            else:
                total_n_all, _, total_M2_all = merge_stats_parallel(all_cat_stats_for_merge)
                if total_n_all > 1: global_std_dev = float(np.sqrt(total_M2_all / (total_n_all - 1)))
                global_std_dev = max(global_std_dev, 1e-6)
            print(f"→ Global std_dev calculated: {global_std_dev:.6f}")
            current_config = {}; category_map = {cat: i for i, cat in enumerate(target_categories)}
            if os.path.exists(main_config_file):
                with open(main_config_file, "r") as f: current_config = json.load(f)
            current_config.update({
                "dataset_name": "quickdraw_sketchrnn", "quickdraw_std_dev": global_std_dev,
                "categories_processed": target_categories, "category_map": category_map,
                "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
                "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 2),
                "max_items_per_category_processed": args.max_items_per_category if args.max_items_per_category else "all",
                "vector_data_path_relative": "quickdraw_vector", 
                "raster_data_path_relative": "quickdraw_raster", 
                "raster_image_size": RASTER_IMG_SIZE,            
                "data_format_note_vector": "Each .npy file: list of 5-element sequences [N,5] (dx,dy,p0,p1,p2), dx,dy normalized.",
                "data_format_note_raster": f"PNG images, {RASTER_IMG_SIZE}x{RASTER_IMG_SIZE}, binary." 
            })
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
            if cat_name not in all_sketches_by_cat_npz: 
                print(f"Warning: No loaded sketch data for {cat_name}, skipping process stage for it.")
                continue
            cat_prog = process_progress.get(cat_name, {})
            if not cat_prog.get("pass1_done", False) or not cat_prog.get("pass2_done", False): 
                tasks_for_processing_stage.append(
                    (cat_name, all_sketches_by_cat_npz[cat_name], 
                     vector_output_base, raster_output_base, 
                     global_std_dev, args.train_ratio, args.val_ratio, 
                     args.threads, RASTER_IMG_SIZE, process_progress) 
                )
        
        if not tasks_for_processing_stage:
            print("→ All target categories already processed (pass1 & pass2 done).")
        else:
            print(f"Processing data for {len(tasks_for_processing_stage)} categories using {args.threads} parallel category workers...")
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
