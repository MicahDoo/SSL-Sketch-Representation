import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import json
from PIL import Image, ImageDraw

# --- Configuration (can be overridden by dataset config or args) ---
IMG_SIZE = 224 
MAX_SEQ_LEN = 70 
DEFAULT_RASTER_PADDING_PERCENT = 0.02 # Default padding for on-the-fly rasterization

class SketchDataset(Dataset):
    def __init__(self, dataset_path, dataset_name="quickdraw", split="train",
                 max_seq_len=MAX_SEQ_LEN, image_size=IMG_SIZE,
                 vector_transform=None, raster_transform=None,
                 config_filename="quickdraw_config.json", # Expected in dataset_path/dataset_name_vector/
                 raster_data_subdir="quickdraw_raster", # Subdirectory for pre-rasterized images
                 on_the_fly_raster_padding_percent=DEFAULT_RASTER_PADDING_PERCENT
                ):
        """
        Args:
            dataset_path (string): Path to the root of processed datasets (e.g., ../processed_data).
            dataset_name (string): Base name of the dataset (e.g., "quickdraw").
                                   Vector data expected in dataset_path/dataset_name_vector/
                                   Raster data expected in dataset_path/dataset_name_raster/
            split (string): "train", "val", or "test".
            max_seq_len (int): Maximum sequence length for vector sketches.
            image_size (int): Size for the rasterized images.
            vector_transform (callable, optional): Optional transform for vector sketch.
            raster_transform (callable, optional): Optional transform for raster image.
            config_filename (string): Name of the JSON config file.
            raster_data_subdir (string): Subdirectory name where pre-rasterized images are stored.
            on_the_fly_raster_padding_percent (float): Padding for on-the-fly rasterization.
        """
        self.dataset_root_path = dataset_path
        self.dataset_name_base = dataset_name.lower()
        self.split = split
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.vector_transform = vector_transform
        self.raster_transform = raster_transform
        self.on_the_fly_raster_padding_percent = on_the_fly_raster_padding_percent

        # Define paths for vector and potential pre-rasterized data
        # Assumes vector data is in a subdir like 'quickdraw_vector'
        self.vector_data_root = os.path.join(self.dataset_root_path, f"{self.dataset_name_base}_vector")
        self.raster_data_root = os.path.join(self.dataset_root_path, raster_data_subdir)
        
        self.processed_sequences = [] # List to store loaded (dx,dy,p_state) sequences
        self.raster_image_paths = []  # List to store paths to pre-rasterized images (or None)
        self.labels = []              # List to store labels
        
        self.config_path = os.path.join(self.vector_data_root, config_filename)
        self.config = {}
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.category_map = self.config.get("category_map", {})
            self.quickdraw_std_dev_for_denorm = self.config.get("quickdraw_std_dev")
            print(f"Loaded config from {self.config_path}")
            if not self.category_map:
                print("Warning: category_map not found or empty in config.")
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Category mapping might be missing.")
            self.category_map = {}
            self.quickdraw_std_dev_for_denorm = None
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {self.config_path}.")
            self.category_map = {}
            self.quickdraw_std_dev_for_denorm = None

        self._load_data()

    def _load_data(self):
        vector_split_path = os.path.join(self.vector_data_root, self.split)
        raster_split_path_base = os.path.join(self.raster_data_root, self.split) # Base for raster categories

        print(f"Loading PREPROCESSED vector data from: {vector_split_path}")
        print(f"Will look for pre-rasterized images in subdirs of: {raster_split_path_base}")

        if not os.path.isdir(vector_split_path):
            print(f"Error: Processed vector data directory not found: {vector_split_path}")
            return

        # Use categories from the config if available and populated, otherwise from vector dir
        if self.config.get("categories_processed"):
            categories_in_split = self.config["categories_processed"]
            # Ensure these categories actually exist as .npy files in the vector_split_path
            categories_in_split = [cat for cat in categories_in_split if os.path.exists(os.path.join(vector_split_path, f"{cat}.npy"))]
            if not categories_in_split: # Fallback if config categories don't match files
                 categories_in_split = sorted([f[:-4] for f in os.listdir(vector_split_path) if f.endswith(".npy")])
        else:
            categories_in_split = sorted([f[:-4] for f in os.listdir(vector_split_path) if f.endswith(".npy")])
        
        if not self.category_map and categories_in_split: # If category_map wasn't in config
            print("Building temporary category_map from directory listing as config map was empty/missing.")
            for i, cat_name in enumerate(categories_in_split):
                self.category_map[cat_name] = i
        
        for category_name in categories_in_split:
            cat_label = self.category_map.get(category_name)
            if cat_label is None:
                print(f"Warning: Category '{category_name}' found but not in category_map. Skipping.")
                continue

            vector_file_path = os.path.join(vector_split_path, f"{category_name}.npy")
            raster_category_dir = os.path.join(raster_split_path_base, category_name)

            try:
                category_vector_sequences = np.load(vector_file_path, allow_pickle=True)
                
                # Get corresponding raster image paths for this category
                # Assumes they are named sketch_00000.png, sketch_00001.png, etc.
                # and their count matches the number of vector sequences.
                potential_raster_files = []
                if os.path.isdir(raster_category_dir):
                    # Sort them to ensure order matches the implicit order in the .npy vector file
                    potential_raster_files = sorted([
                        os.path.join(raster_category_dir, f)
                        for f in os.listdir(raster_category_dir) if f.startswith("sketch_") and f.endswith(".png")
                    ])

                for i, seq_array in enumerate(category_vector_sequences):
                    if isinstance(seq_array, np.ndarray) and seq_array.ndim == 2 and seq_array.shape[1] == 5:
                        self.processed_sequences.append(seq_array)
                        self.labels.append(cat_label)
                        
                        # Check for corresponding pre-rasterized image
                        if i < len(potential_raster_files):
                            self.raster_image_paths.append(potential_raster_files[i])
                        else:
                            self.raster_image_paths.append(None) # Mark that it needs on-the-fly generation
                    else:
                        print(f"Warning: Skipping invalid vector item in {vector_file_path}. Expected NumPy array (N,5).")
            except Exception as e:
                print(f"Error loading or processing file {vector_file_path}: {e}")
        
        if not self.processed_sequences:
            print(f"Warning: No processed sequences loaded for {self.dataset_name_base} {self.split}.")
        else:
            print(f"Loaded {len(self.processed_sequences)} vector sequences for {self.dataset_name_base} {self.split}.")
            num_prerasterized = sum(1 for p in self.raster_image_paths if p is not None)
            print(f"  Found {num_prerasterized} corresponding pre-rasterized image paths.")


    def _vector_to_raster(self, normalized_vector_sequence_np, image_size, 
                          line_thickness_raster=2, padding_percent_raster=None):
        """
        Rasterizes a NORMALIZED delta vector sequence to a PIL Image.
        This is the on-the-fly version.
        """
        if padding_percent_raster is None:
            padding_percent_raster = self.on_the_fly_raster_padding_percent

        image = Image.new("L", (image_size, image_size), "white")
        draw = ImageDraw.Draw(image)
        abs_segments = [] 
        current_abs_x, current_abs_y = 0.0, 0.0 

        for i in range(normalized_vector_sequence_np.shape[0]):
            dx, dy, p_down, p_up, p_eos = normalized_vector_sequence_np[i]
            next_abs_x, next_abs_y = current_abs_x + dx, current_abs_y + dy
            if p_down > 0.5: # Pen is drawing
                 if i > 0 : # Not the very first point (which is effectively a move_to)
                     abs_segments.append(((current_abs_x, current_abs_y), (next_abs_x, next_abs_y)))
            current_abs_x, current_abs_y = next_abs_x, next_abs_y
            if p_eos > 0.5: break 

        if not abs_segments: return image.convert("1")

        all_x_coords = [p[0][0] for p in abs_segments] + [p[1][0] for p in abs_segments]
        all_y_coords = [p[0][1] for p in abs_segments] + [p[1][1] for p in abs_segments]
        if not all_x_coords or not all_y_coords: return image.convert("1")

        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        sketch_width = max_x - min_x
        sketch_height = max_y - min_y

        if sketch_width < 1e-6 and sketch_height < 1e-6:
            if abs_segments:
                 pt_x = (abs_segments[0][0][0] - min_x) 
                 pt_y = (abs_segments[0][0][1] - min_y) 
                 draw_x = image_size / 2 + pt_x 
                 draw_y = image_size / 2 + pt_y
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
        if idx >= len(self.processed_sequences): # Basic bounds check
            # This case should ideally not be hit if __len__ is correct and used properly by DataLoader
            print(f"Warning: Index {idx} out of bounds. Max index: {len(self.processed_sequences)-1}")
            # Return a dummy item to prevent crash, though this indicates a deeper issue
            dummy_vec = torch.zeros((self.max_seq_len, 5), dtype=torch.float32)
            dummy_img = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            return {"vector_sketch": dummy_vec, "raster_image": dummy_img, 
                    "attention_mask": dummy_mask, "label": torch.tensor(-1, dtype=torch.long)}

        processed_normalized_sequence_np = self.processed_sequences[idx].copy() 
        label_val = self.labels[idx]
        raster_path = self.raster_image_paths[idx]
        
        # --- Vector Sketch Processing ---
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

        # --- Raster Image Processing ---
        pil_image = None
        if raster_path and os.path.exists(raster_path):
            try:
                pil_image = Image.open(raster_path).convert('L') # Ensure grayscale 'L' before '1'
                if pil_image.size != (self.image_size, self.image_size):
                    pil_image = pil_image.resize((self.image_size, self.image_size), Image.BILINEAR)
                pil_image = pil_image.convert('1') # Convert to binary
            except Exception as e:
                print(f"Warning: Error loading pre-rasterized image {raster_path}: {e}. Falling back to on-the-fly.")
                pil_image = None
        
        if pil_image is None: # Fallback to on-the-fly rasterization
            # print(f"Note: Generating raster image on-the-fly for index {idx}") # For debugging
            pil_image = self._vector_to_raster(processed_normalized_sequence_np, self.image_size)
            
        # Apply raster transform (usually ToTensor and possibly Normalize)
        current_raster_transform = self.raster_transform
        if current_raster_transform is None:
            current_raster_transform = transforms.ToTensor() # Default to ToTensor if none provided
        
        raster_image_tensor = current_raster_transform(pil_image)
        # Ensure tensor is [1, H, W] if ToTensor converts '1' mode PIL to [H,W]
        if raster_image_tensor.ndim == 2:
            raster_image_tensor = raster_image_tensor.unsqueeze(0)


        label = torch.tensor(label_val, dtype=torch.long)
        
        return {
            "vector_sketch": sketch_tensor, 
            "raster_image": raster_image_tensor,
            "attention_mask": attention_mask, 
            "label": label
        }

