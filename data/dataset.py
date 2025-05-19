import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import json
from PIL import Image, ImageDraw
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor # For parallel loading

# --- Configuration (can be overridden by dataset config or args) ---
IMG_SIZE = 224 
MAX_SEQ_LEN = 70 
DEFAULT_RASTER_PADDING_PERCENT = 0.02 

class SketchDataset(Dataset):
    def __init__(self, dataset_path, dataset_name="quickdraw", split="train",
                 max_seq_len=MAX_SEQ_LEN, image_size=IMG_SIZE,
                 vector_transform=None, raster_transform=None,
                 config_filename="quickdraw_config.json", 
                 raster_data_subdir="quickdraw_raster", 
                 on_the_fly_raster_padding_percent=DEFAULT_RASTER_PADDING_PERCENT,
                 num_initial_load_workers=4 # New parameter for parallel loading
                ):
        self.dataset_root_path = dataset_path
        self.dataset_name_base = dataset_name.lower()
        self.split = split
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.vector_transform = vector_transform
        self.raster_transform = raster_transform
        self.on_the_fly_raster_padding_percent = on_the_fly_raster_padding_percent
        self.num_initial_load_workers = max(1, num_initial_load_workers) # Ensure at least 1 worker

        self.vector_data_root = os.path.join(self.dataset_root_path, f"{self.dataset_name_base}_vector")
        self.raster_data_root = os.path.join(self.dataset_root_path, raster_data_subdir)
        
        self.processed_sequences = [] 
        self.raster_image_paths = []  
        self.labels = []              
        
        self.config_path = os.path.join(self.vector_data_root, config_filename)
        self.config = {}
        print(f"DEBUG [SketchDataset]: Attempting to load config from: {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.category_map = self.config.get("category_map", {})
            self.quickdraw_std_dev_for_denorm = self.config.get("quickdraw_std_dev")
            print(f"DEBUG [SketchDataset]: Loaded config. Category_map items: {len(self.category_map)}. Std_dev: {self.quickdraw_std_dev_for_denorm}")
            if not self.category_map:
                print("DEBUG [SketchDataset]: Warning: category_map not found or empty in config.")
        except FileNotFoundError:
            print(f"DEBUG [SketchDataset]: Warning: Config file not found at {self.config_path}. Category mapping might be missing.")
            self.category_map = {}
            self.quickdraw_std_dev_for_denorm = None
        except json.JSONDecodeError:
            print(f"DEBUG [SketchDataset]: Warning: Error decoding JSON from {self.config_path}.")
            self.category_map = {}
            self.quickdraw_std_dev_for_denorm = None

        self._load_data()

    def _load_single_category_data(self, category_info_tuple):
        """Loads data for a single category."""
        category_name, cat_label, vector_split_path, raster_split_path_base = category_info_tuple
        
        local_processed_sequences = []
        local_labels = []
        local_raster_image_paths = []

        vector_file_path = os.path.join(vector_split_path, f"{category_name}.npy")
        raster_category_dir = os.path.join(raster_split_path_base, category_name)

        if not os.path.exists(vector_file_path):
            # This print can be verbose if many categories are missing their .npy files
            # print(f"DEBUG [SketchDataset._load_single_category_data]: Vector file not found for {category_name}: {vector_file_path}. Skipping category.")
            return [], [], [] # Return empty lists for this category
            
        try:
            category_vector_sequences = np.load(vector_file_path, allow_pickle=True)
            potential_raster_files = []
            if os.path.isdir(raster_category_dir):
                potential_raster_files = sorted([
                    os.path.join(raster_category_dir, f)
                    for f in os.listdir(raster_category_dir) if f.startswith("sketch_") and f.endswith(".png")
                ])

            for i, seq_array in enumerate(category_vector_sequences):
                if isinstance(seq_array, np.ndarray) and seq_array.ndim == 2 and seq_array.shape[1] == 5:
                    local_processed_sequences.append(seq_array)
                    local_labels.append(cat_label)
                    
                    if i < len(potential_raster_files) and os.path.exists(potential_raster_files[i]):
                        local_raster_image_paths.append(potential_raster_files[i])
                    else:
                        local_raster_image_paths.append(None) 
                # else:
                    # print(f"DEBUG [SketchDataset._load_data]: Warning: Skipping invalid vector item in {vector_file_path}.")
            return local_processed_sequences, local_labels, local_raster_image_paths
        except Exception as e:
            print(f"DEBUG [SketchDataset._load_single_category_data]: Error loading/processing file {vector_file_path}: {e}")
            return [], [], []


    def _load_data(self):
        vector_split_path = os.path.join(self.vector_data_root, self.split)
        raster_split_path_base = os.path.join(self.raster_data_root, self.split) 

        print(f"DEBUG [SketchDataset._load_data]: Loading PREPROCESSED vector data from: {vector_split_path}")
        print(f"DEBUG [SketchDataset._load_data]: Will look for pre-rasterized images in subdirs of: {raster_split_path_base}")

        if not os.path.isdir(vector_split_path):
            print(f"DEBUG [SketchDataset._load_data]: Error: Processed vector data directory not found: {vector_split_path}")
            return

        categories_in_split_from_fs = []
        if os.path.exists(vector_split_path):
            categories_in_split_from_fs = sorted([f[:-4] for f in os.listdir(vector_split_path) if f.endswith(".npy")])
        
        categories_to_iterate = []
        if self.config.get("categories_processed"):
            categories_to_iterate_from_config = self.config["categories_processed"]
            categories_to_iterate = [cat for cat in categories_to_iterate_from_config if os.path.exists(os.path.join(vector_split_path, f"{cat}.npy"))]
            if not categories_to_iterate and categories_to_iterate_from_config: 
                 print("DEBUG [SketchDataset._load_data]: Config categories_processed led to empty list after file check. Falling back to FS scan.")
                 categories_to_iterate = categories_in_split_from_fs
            elif not categories_to_iterate_from_config: # If config key exists but list is empty
                 categories_to_iterate = categories_in_split_from_fs

        else: # Config doesn't have 'categories_processed' or it's None
            categories_to_iterate = categories_in_split_from_fs
        
        if not self.category_map and categories_to_iterate: 
            print("DEBUG [SketchDataset._load_data]: Building temporary category_map from directory listing as config map was empty/missing.")
            for i, cat_name in enumerate(categories_to_iterate):
                self.category_map[cat_name] = i 
        
        if not categories_to_iterate:
            print(f"DEBUG [SketchDataset._load_data]: No categories found to process for split '{self.split}'.")
            return

        print(f"DEBUG [SketchDataset._load_data]: Preparing to load data for {len(categories_to_iterate)} categories using {self.num_initial_load_workers} worker(s).")

        tasks = []
        for category_name in categories_to_iterate:
            cat_label = self.category_map.get(category_name)
            if cat_label is None:
                print(f"DEBUG [SketchDataset._load_data]: Warning: Category '{category_name}' in iteration list but not in category_map. Skipping.")
                continue
            # We pass paths to the worker, not category_name for os.listdir inside worker
            tasks.append((category_name, cat_label, vector_split_path, raster_split_path_base))

        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_initial_load_workers) as executor:
            # tqdm wraps the executor.map call
            all_results = list(tqdm(executor.map(self._load_single_category_data, tasks), 
                               total=len(tasks), 
                               desc=f"Loading category data for split '{self.split}'"))

        # Consolidate results from all threads
        for cat_vectors, cat_labels, cat_raster_paths in all_results:
            self.processed_sequences.extend(cat_vectors)
            self.labels.extend(cat_labels)
            self.raster_image_paths.extend(cat_raster_paths)
        
        if not self.processed_sequences:
            print(f"DEBUG [SketchDataset._load_data]: Warning: No processed sequences loaded for {self.dataset_name_base} {self.split} after attempting all categories.")
        else:
            print(f"DEBUG [SketchDataset._load_data]: Loaded {len(self.processed_sequences)} total vector sequences for {self.dataset_name_base} {self.split}.")
            num_prerasterized = sum(1 for p in self.raster_image_paths if p is not None)
            print(f"DEBUG [SketchDataset._load_data]:   Found {num_prerasterized} corresponding pre-rasterized image paths.")


    def _vector_to_raster(self, normalized_vector_sequence_np, image_size, 
                          line_thickness_raster=2, padding_percent_raster=None):
        if padding_percent_raster is None:
            padding_percent_raster = self.on_the_fly_raster_padding_percent
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

    def __len__(self):
        return len(self.processed_sequences)

    def __getitem__(self, idx):
        if idx >= len(self.processed_sequences): 
            print(f"DEBUG [SketchDataset.__getitem__]: Warning: Index {idx} out of bounds. Max index: {len(self.processed_sequences)-1}")
            dummy_vec = torch.zeros((self.max_seq_len, 5), dtype=torch.float32)
            dummy_img = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            return {"vector_sketch": dummy_vec, "raster_image": dummy_img, 
                    "attention_mask": dummy_mask, "label": torch.tensor(-1, dtype=torch.long)}

        processed_normalized_sequence_np = self.processed_sequences[idx].copy() 
        label_val = self.labels[idx]
        raster_path = self.raster_image_paths[idx]
        
        seq_len = processed_normalized_sequence_np.shape[0]
        sketch_tensor = torch.tensor(processed_normalized_sequence_np, dtype=torch.float32)
        if seq_len < self.max_seq_len:
            padding_tensor = torch.zeros((self.max_seq_len - seq_len, 5), dtype=torch.float32)
            sketch_tensor = torch.cat([sketch_tensor, padding_tensor], dim=0)
        elif seq_len > self.max_seq_len:
            sketch_tensor = sketch_tensor[:self.max_seq_len, :]
            if sketch_tensor[-1, 4] != 1: 
                sketch_tensor[-1, 0:2] = 0.0; sketch_tensor[-1, 2:4] = 0.0; sketch_tensor[-1, 4] = 1.0 
        
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        actual_len = min(seq_len, self.max_seq_len)
        attention_mask[:actual_len] = 1

        pil_image = None
        if raster_path and os.path.exists(raster_path):
            try:
                pil_image = Image.open(raster_path).convert('L') 
                if pil_image.size != (self.image_size, self.image_size):
                    pil_image = pil_image.resize((self.image_size, self.image_size), Image.BILINEAR)
                pil_image = pil_image.convert('1') 
            except Exception as e:
                pil_image = None
        
        if pil_image is None: 
            pil_image = self._vector_to_raster(processed_normalized_sequence_np, self.image_size)
            
        current_raster_transform = self.raster_transform
        if current_raster_transform is None:
            current_raster_transform = transforms.ToTensor() 
        
        raster_image_tensor = current_raster_transform(pil_image)
        if raster_image_tensor.ndim == 2:
            raster_image_tensor = raster_image_tensor.unsqueeze(0)

        label = torch.tensor(label_val, dtype=torch.long)
        
        return {
            "vector_sketch": sketch_tensor, 
            "raster_image": raster_image_tensor,
            "attention_mask": attention_mask, 
            "label": label
        }

