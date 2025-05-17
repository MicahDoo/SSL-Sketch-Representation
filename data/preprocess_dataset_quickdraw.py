# data/preprocess_dataset_quickdraw.py
import os
import json
import random
import argparse
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw 
import multiprocessing

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
VAL_RATIO = 0.10 
RASTER_IMG_SIZE = 224 
CONFIG_FILENAME = "quickdraw_config.json"

# --- Helper Functions --- 
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
        # This message is now more of a fallback, as main() should pre-filter
        print(f"File '{file_name}' not found locally and --skip_raw_download is set. Download will be skipped.")
        return target_path, False

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, 'wb') as f, tqdm(
            desc=f"Downloading {category_name[:20]}...",
            total=int(response.headers.get('content-length', 0)),
            unit='iB', unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk: f.write(chunk); bar.update(len(chunk))
        return target_path, True
    except requests.exceptions.RequestException:
        if os.path.exists(target_path): os.remove(target_path)
        return target_path, False

def load_simplified_drawings(ndjson_path, max_items_per_category=None):
    drawings = []
    try:
        with open(ndjson_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_items_per_category is not None and i >= max_items_per_category: break
                try:
                    obj = json.loads(line)
                    if obj.get('recognized', False): drawings.append(obj['drawing'])
                except (json.JSONDecodeError, KeyError): pass
    except FileNotFoundError: 
        # This case should be less frequent if main() pre-filters when skip_raw_download is True
        print(f"Warning: File not found during load: {ndjson_path}. Skipping category.")
        return [] 
    except Exception as e: print(f"Error reading file {ndjson_path}: {e}")
    return drawings

def convert_raw_sketch_to_delta_sequence_mp_helper(raw_sketch_strokes):
    return convert_raw_sketch_to_delta_sequence(raw_sketch_strokes)

def convert_raw_sketch_to_delta_sequence(raw_sketch_strokes):
    points = []
    last_x, last_y = 0, 0
    num_strokes = len(raw_sketch_strokes)
    if num_strokes == 0: return np.zeros((0,5), dtype=np.float32)
    for stroke_idx, stroke in enumerate(raw_sketch_strokes):
        xs, ys = stroke
        if not xs or len(xs) != len(ys): continue 
        dx0 = xs[0] - last_x; dy0 = ys[0] - last_y
        if stroke_idx > 0: points.append([dx0, dy0, 0, 1, 0]); dx0, dy0 = 0, 0
        is_last_point_in_sketch = (stroke_idx == num_strokes - 1) and (len(xs) == 1)
        p_state = [0,0,1] if is_last_point_in_sketch else [1,0,0]
        points.append([dx0, dy0] + p_state)
        last_x, last_y = xs[0], ys[0]
        for i in range(1, len(xs)):
            dx = xs[i] - last_x; dy = ys[i] - last_y
            last_x, last_y = xs[i], ys[i]
            is_last_point_in_sketch = (stroke_idx == num_strokes - 1) and (i == len(xs) - 1)
            p_state = [0,0,1] if is_last_point_in_sketch else [1,0,0]
            points.append([dx, dy] + p_state)
        if stroke_idx < num_strokes - 1:
            if points and points[-1][4] == 0: 
                points[-1][2] = 0; points[-1][3] = 1
    if points: 
        points[-1][2] = 0; points[-1][3] = 0; points[-1][4] = 1 
    return np.array(points, dtype=np.float32)

def rasterize_sequence_to_pil_image(normalized_vector_sequence_np, image_size, line_thickness_raster=2, padding_percent_raster=0.02):
    image = Image.new("L", (image_size, image_size), "white") 
    draw = ImageDraw.Draw(image)
    abs_segments = [] 
    current_abs_x, current_abs_y = 0.0, 0.0
    for i in range(normalized_vector_sequence_np.shape[0]):
        dx, dy, p_down, p_up, p_eos = normalized_vector_sequence_np[i]
        next_abs_x, next_abs_y = current_abs_x + dx, current_abs_y + dy
        if p_down > 0.5:
             if i > 0 : 
                 abs_segments.append(((current_abs_x, current_abs_y), (next_abs_x, next_abs_y)))
        current_abs_x, current_abs_y = next_abs_x, next_abs_y
        if p_eos > 0.5: break
    if not abs_segments: return image.convert("1")
    all_x_coords = [p[0][0] for p in abs_segments] + [p[1][0] for p in abs_segments]
    all_y_coords = [p[0][1] for p in abs_segments] + [p[1][1] for p in abs_segments]
    if not all_x_coords or not all_y_coords: return image.convert("1")
    min_x, max_x = min(all_x_coords), max(all_x_coords)
    min_y, max_y = min(all_y_coords), max(all_y_coords)
    sketch_width = max_x - min_x; sketch_height = max_y - min_y
    if sketch_width < 1e-6 and sketch_height < 1e-6: 
        if abs_segments:
             pt_x = (abs_segments[0][0][0] - min_x); pt_y = (abs_segments[0][0][1] - min_y) 
             draw_x = image_size / 2 + pt_x; draw_y = image_size / 2 + pt_y
             draw.ellipse([(draw_x-1, draw_y-1), (draw_x+1, draw_y+1)], fill="black")
        return image.convert("1")
    canvas_draw_area_width = image_size * (1 - 2 * padding_percent_raster)
    canvas_draw_area_height = image_size * (1 - 2 * padding_percent_raster)
    scale_factor = 1.0
    if sketch_width > 1e-6: scale_factor = canvas_draw_area_width / sketch_width
    if sketch_height > 1e-6: scale_factor = min(scale_factor, canvas_draw_area_height / sketch_height)
    offset_x_canvas = (image_size - (sketch_width * scale_factor)) / 2.0
    offset_y_canvas = (image_size - (sketch_height * scale_factor)) / 2.0
    for p1_abs, p2_abs in abs_segments:
        x1 = ((p1_abs[0] - min_x) * scale_factor) + offset_x_canvas
        y1 = ((p1_abs[1] - min_y) * scale_factor) + offset_y_canvas
        x2 = ((p2_abs[0] - min_x) * scale_factor) + offset_x_canvas
        y2 = ((p2_abs[1] - min_y) * scale_factor) + offset_y_canvas
        draw.line([(x1, y1), (x2, y2)], fill="black", width=line_thickness_raster)
    del draw
    return image.convert("1")

def normalize_and_rasterize_mp_helper(args_tuple):
    seq_array, std_dev_val, raster_img_size_val = args_tuple
    norm_seq = seq_array.copy()
    norm_seq[:, :2] /= std_dev_val
    pil_img = rasterize_sequence_to_pil_image(norm_seq, raster_img_size_val)
    return norm_seq, pil_img

def main(categories_to_process_initial, raw_dir_param, processed_dir_base_param, train_r, val_r, max_items_cat, skip_raw_download_flag, num_workers_preprocess):
    print(f"--- Starting QuickDraw Preprocessing for initially {len(categories_to_process_initial)} target categories ---")
    if skip_raw_download_flag:
        print("!!! --skip_raw_download flag is active. Will only process locally existing .ndjson files. !!!")
    if max_items_cat: print(f"Max items per category: {max_items_cat}.")
    
    pool_processes = None 
    actual_workers_to_be_used = 0
    if num_workers_preprocess is not None and num_workers_preprocess <= 0:
        pool_processes = 1; actual_workers_to_be_used = 1
        print(f"Using {actual_workers_to_be_used} worker process for CPU-bound tasks (num_workers_preprocess <= 0).")
    elif num_workers_preprocess is not None:
        pool_processes = num_workers_preprocess; actual_workers_to_be_used = pool_processes
        print(f"Using {actual_workers_to_be_used} worker processes for CPU-bound tasks.")
    else: 
        try:
            actual_workers_to_be_used = os.cpu_count()
            print(f"Using default number of worker processes (os.cpu_count() = {actual_workers_to_be_used}) for CPU-bound tasks.")
        except NotImplementedError:
            print("Warning: os.cpu_count() is not available. Defaulting to 1 worker process.")
            pool_processes = 1; actual_workers_to_be_used = 1

    vector_proc_base = os.path.join(processed_dir_base_param, "quickdraw_vector")
    raster_proc_base = os.path.join(processed_dir_base_param, "quickdraw_raster")
    raw_ndjson_dir_path = os.path.join(raw_dir_param, "quickdraw_raw") # Define path to raw .ndjson files
    
    os.makedirs(vector_proc_base, exist_ok=True)
    os.makedirs(raster_proc_base, exist_ok=True)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(vector_proc_base, split), exist_ok=True)
        os.makedirs(os.path.join(raster_proc_base, split), exist_ok=True)
        # Category subdirs for raster images are created later, only for categories with data

    # --- Determine effective categories to process ---
    categories_to_actually_process = []
    tqdm_desc_cat_load = "Checking/Downloading Categories"
    if skip_raw_download_flag:
        print(f"Scanning {raw_ndjson_dir_path} for existing .ndjson files...")
        tqdm_desc_cat_load = "Identifying Local Categories"
        if os.path.isdir(raw_ndjson_dir_path):
            existing_ndjson_files = {f for f in os.listdir(raw_ndjson_dir_path) if f.endswith(".ndjson")}
            for cat_name in categories_to_process_initial:
                if f"{cat_name}.ndjson" in existing_ndjson_files:
                    categories_to_actually_process.append(cat_name)
            if not categories_to_actually_process and categories_to_process_initial:
                print(f"Warning: --skip_raw_download is set, but no .ndjson files found in {raw_ndjson_dir_path} for the target categories.")
            elif categories_to_actually_process:
                 print(f"Found {len(categories_to_actually_process)} categories locally. Will process only these.")
        else:
            print(f"Warning: --skip_raw_download is set, but raw directory {raw_ndjson_dir_path} not found.")
        if not categories_to_actually_process: # If still empty, means no local files for target list
             print("No local files found for the specified categories. No categories will be processed.")
             return # Exit if no categories to process
    else:
        categories_to_actually_process = categories_to_process_initial

    if not categories_to_actually_process:
        print("No categories to process. Exiting.")
        return
    
    print(f"Final list of categories to process: {len(categories_to_actually_process)}")
    # Create raster category subdirs only for categories that will actually be processed
    for split in ("train", "val", "test"):
        for cat_name in categories_to_actually_process:
             os.makedirs(os.path.join(raster_proc_base, split, cat_name), exist_ok=True)


    all_sketches_by_cat = {}
    category_map = {}
        
    for idx, cat_name in enumerate(tqdm(categories_to_actually_process, desc=tqdm_desc_cat_load)):
        category_map[cat_name] = idx # Map based on the final list being processed
        ndjson_path, ok = download_category_file(cat_name, raw_dir_param, skip_actual_download=skip_raw_download_flag)
        if not ok: 
            # This should ideally not happen if skip_raw_download_flag is True because categories_to_actually_process was pre-filtered
            print(f"  -> Critical: Skipping '{cat_name}' (file check failed unexpectedly or download failed).")
            continue
        
        drawings = load_simplified_drawings(ndjson_path, max_items_per_category=max_items_cat)
        if drawings: all_sketches_by_cat[cat_name] = drawings
        else: print(f"  -> No valid drawings loaded for '{cat_name}'.")

    # Check if any categories were successfully loaded before proceeding
    if not all_sketches_by_cat:
        print("No sketches loaded for any category. Aborting further processing.")
        return

    print("\n--- Pass 1: Converting & Calculating StdDev ---")
    # ... (rest of Pass 1 and Pass 2 logic remains the same as in previous version) ...
    all_train_candidate_deltas = []
    all_processed_sketches_by_cat = {}
    
    if pool_processes == 1: 
        print("  Executing Pass 1 conversions in a single process.")
        for cat_name, raw_drawings in tqdm(all_sketches_by_cat.items(), desc="Pass 1 Processing (Single Proc)"):
            processed_sequences_for_cat = [seq for sketch_drawing in raw_drawings 
                                           if (seq := convert_raw_sketch_to_delta_sequence(sketch_drawing)).size > 0]
            if not processed_sequences_for_cat: print(f"No valid sequences for '{cat_name}'."); continue
            all_processed_sketches_by_cat[cat_name] = processed_sequences_for_cat
            random.shuffle(processed_sequences_for_cat)
            n_train_cat = int(train_r * len(processed_sequences_for_cat))
            for seq_array in processed_sequences_for_cat[:n_train_cat]:
                all_train_candidate_deltas.extend(seq_array[:,0]); all_train_candidate_deltas.extend(seq_array[:,1])
    else: 
        print(f"  Executing Pass 1 conversions with a pool of {actual_workers_to_be_used} processes.")
        with multiprocessing.Pool(processes=pool_processes) as pool: 
            for cat_name, raw_drawings in tqdm(all_sketches_by_cat.items(), desc="Pass 1 Processing (Multi Proc)"):
                results = pool.map(convert_raw_sketch_to_delta_sequence_mp_helper, raw_drawings)
                processed_sequences_for_cat = [seq for seq in results if seq.size > 0]
                if not processed_sequences_for_cat: print(f"No valid sequences for '{cat_name}'."); continue
                all_processed_sketches_by_cat[cat_name] = processed_sequences_for_cat
                random.shuffle(processed_sequences_for_cat)
                n_train_cat = int(train_r * len(processed_sequences_for_cat))
                for seq_array in processed_sequences_for_cat[:n_train_cat]:
                    all_train_candidate_deltas.extend(seq_array[:,0]); all_train_candidate_deltas.extend(seq_array[:,1])

    if not all_train_candidate_deltas: std_dev = 1.0; print("Warning: No deltas; std_dev=1.0")
    else: std_dev = float(np.std(all_train_candidate_deltas)); std_dev = max(std_dev, 1e-6) 
    print(f"Global std_dev = {std_dev:.4f}")

    print("\n--- Pass 2: Normalizing, Rasterizing, Splitting, Saving ---")
    
    if pool_processes == 1:
        print("  Executing Pass 2 (Normalize & Rasterize) in a single process.")
        for cat_name, processed_sequences in tqdm(all_processed_sketches_by_cat.items(), desc="Pass 2 (Single Proc)"):
            norm_and_raster_results = []
            for seq_array in processed_sequences:
                norm_seq, pil_img = normalize_and_rasterize_mp_helper((seq_array, std_dev, RASTER_IMG_SIZE))
                norm_and_raster_results.append((norm_seq, pil_img))
            
            normalized_vector_sequences, raster_images_for_cat = zip(*norm_and_raster_results) if norm_and_raster_results else ([], [])
            combined = list(zip(normalized_vector_sequences, raster_images_for_cat))
            random.shuffle(combined)
            shuffled_vectors, shuffled_rasters = zip(*combined) if combined else ([], [])
            n_total = len(shuffled_vectors)
            n_train = int(train_r * n_total); n_val = int(val_r * n_total)
            splits_vector = {"train": list(shuffled_vectors[:n_train]), "val": list(shuffled_vectors[n_train : n_train + n_val]), "test": list(shuffled_vectors[n_train + n_val:])}
            splits_raster = {"train": list(shuffled_rasters[:n_train]), "val": list(shuffled_rasters[n_train : n_train + n_val]), "test": list(shuffled_rasters[n_train + n_val:])}
            for split_name in ["train", "val", "test"]:
                vector_split_data = splits_vector[split_name]; raster_split_data = splits_raster[split_name]
                if not vector_split_data: continue
                vector_out_path = os.path.join(vector_proc_base, split_name, f"{cat_name}.npy")
                np.save(vector_out_path, np.array(vector_split_data, dtype=object))
                raster_cat_split_dir = os.path.join(raster_proc_base, split_name, cat_name)
                for i, pil_img in enumerate(raster_split_data):
                    img_filename = f"sketch_{i:05d}.png"
                    pil_img.save(os.path.join(raster_cat_split_dir, img_filename))
    else: 
        print(f"  Executing Pass 2 (Normalize & Rasterize) with a pool of {actual_workers_to_be_used} processes.")
        with multiprocessing.Pool(processes=pool_processes) as pool: 
            for cat_name, processed_sequences in tqdm(all_processed_sketches_by_cat.items(), desc="Pass 2 (Multi Proc)"):
                tasks = [(seq, std_dev, RASTER_IMG_SIZE) for seq in processed_sequences]
                norm_and_raster_results = pool.map(normalize_and_rasterize_mp_helper, tasks)
                normalized_vector_sequences, raster_images_for_cat = zip(*norm_and_raster_results) if norm_and_raster_results else ([], [])
                combined = list(zip(normalized_vector_sequences, raster_images_for_cat))
                random.shuffle(combined)
                shuffled_vectors, shuffled_rasters = zip(*combined) if combined else ([], [])
                n_total = len(shuffled_vectors)
                n_train = int(train_r * n_total); n_val = int(val_r * n_total)
                splits_vector = {"train": list(shuffled_vectors[:n_train]), "val": list(shuffled_vectors[n_train : n_train + n_val]), "test": list(shuffled_vectors[n_train + n_val:])}
                splits_raster = {"train": list(shuffled_rasters[:n_train]), "val": list(shuffled_rasters[n_train : n_train + n_val]), "test": list(shuffled_rasters[n_train + n_val:])}
                for split_name in ["train", "val", "test"]:
                    vector_split_data = splits_vector[split_name]; raster_split_data = splits_raster[split_name]
                    if not vector_split_data: continue
                    vector_out_path = os.path.join(vector_proc_base, split_name, f"{cat_name}.npy")
                    np.save(vector_out_path, np.array(vector_split_data, dtype=object))
                    raster_cat_split_dir = os.path.join(raster_proc_base, split_name, cat_name)
                    for i, pil_img in enumerate(raster_split_data):
                        img_filename = f"sketch_{i:05d}.png"
                        pil_img.save(os.path.join(raster_cat_split_dir, img_filename))

    config_data = {
        "dataset_name": "quickdraw", "quickdraw_std_dev": std_dev, "category_map": category_map, # category_map is now based on actually processed categories
        "categories_processed": categories_to_actually_process, # Store the list of categories that were actually processed
        "train_ratio": train_r, "val_ratio": val_r,
        "test_ratio": round(1.0 - train_r - val_r, 2), 
        "max_items_per_category_processed": max_items_cat if max_items_cat else "all",
        "vector_data_path_relative": "quickdraw_vector", 
        "raster_data_path_relative": "quickdraw_raster", 
        "raster_image_size": RASTER_IMG_SIZE,
        "data_format_note_vector": "Each .npy file in quickdraw_vector: list of sequences [N,5] (dx,dy,p0,p1,p2), dx,dy normalized.",
        "data_format_note_raster": f"PNG images in quickdraw_raster, {RASTER_IMG_SIZE}x{RASTER_IMG_SIZE}, binary."
    }
    config_file_path = os.path.join(vector_proc_base, CONFIG_FILENAME) 
    with open(config_file_path, 'w') as f: json.dump(config_data, f, indent=4)
    print(f"\nPreprocessing complete. Vector and Raster data saved. Config: {config_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuickDraw Preprocessing (Vector & Raster).")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--all_categories", action="store_true")
    parser.add_argument("--skip_raw_download", action="store_true", 
                        help="Skip attempting to download raw .ndjson files. Only processes locally existing files from the target categories.")
    parser.add_argument("--raw_dir", default=RAW_DOWNLOAD_DIR_ROOT, 
                        help=f"Dir for raw .ndjson files (default: {RAW_DOWNLOAD_DIR_ROOT})")
    parser.add_argument("--processed_dir", default=PROCESSED_DATA_DIR_ROOT, 
                        help=f"Base directory to save processed data (default: {PROCESSED_DATA_DIR_ROOT})")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--max_items_per_category", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of worker processes for CPU-bound tasks. Default: all available CPUs. Set to 0 or 1 for single process.")

    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    print(f"Using Raw Directory: {os.path.abspath(args.raw_dir)}")
    print(f"Using Processed Directory: {os.path.abspath(args.processed_dir)}")

    categories_to_process_initial_list = [] # Renamed to avoid confusion
    if args.all_categories:
        print("Attempting to download the full list of categories...")
        downloaded_categories = download_all_categories_list()
        if downloaded_categories: categories_to_process_initial_list = downloaded_categories
        else: print("Failed to download category list. Using smaller fallback list."); categories_to_process_initial_list = FALLBACK_CATEGORIES
        print(f"Targeting {len(categories_to_process_initial_list)} QuickDraw categories.")
    elif args.categories:
        categories_to_process_initial_list = sorted(list(set(args.categories))) 
        print(f"Targeting specified categories: {categories_to_process_initial_list}")
    else:
        categories_to_process_initial_list = sorted(DEFAULT_CATEGORIES) 
        print(f"Targeting DEFAULT categories: {categories_to_process_initial_list}")
    
    main(
        categories_to_process_initial=categories_to_process_initial_list, # Pass the initial target list
        raw_dir_param=args.raw_dir, 
        processed_dir_base_param=args.processed_dir, 
        train_r=args.train_ratio,
        val_r=args.val_ratio,
        max_items_cat=args.max_items_per_category,
        skip_raw_download_flag=args.skip_raw_download,
        num_workers_preprocess=args.num_workers 
    )
