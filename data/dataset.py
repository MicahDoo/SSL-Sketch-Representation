import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import json # For loading config if needed for category_map
from PIL import Image, ImageDraw

# --- Configuration (can be overridden by dataset config or args) ---
IMG_SIZE = 224 
MAX_SEQ_LEN = 70 # Default, should align with how data was preprocessed or a new target
# LATENT_DIM is a model property, not dataset directly

class SketchDataset(Dataset):
    def __init__(self, dataset_path, dataset_name="quickdraw", split="train",
                 max_seq_len=MAX_SEQ_LEN, image_size=IMG_SIZE,
                 vector_transform=None, raster_transform=None,
                 # categories=None # No longer needed if using all from split dir
                 config_filename="quickdraw_config.json"): # Name of the config file
        """
        Args:
            dataset_path (string): Path to the root of processed datasets (e.g., ../processed_data).
            dataset_name (string): Name of the dataset ("quickdraw").
            split (string): "train", "val", or "test".
            max_seq_len (int): Maximum sequence length for vector sketches (for padding/truncation).
            image_size (int): Size for the rasterized images.
            vector_transform (callable, optional): Optional transform for vector sketch (rarely used now).
            raster_transform (callable, optional): Optional transform for raster image.
            config_filename (string): Name of the JSON config file within dataset_name directory.
        """
        self.dataset_root_path = dataset_path # e.g., ../processed_data
        self.dataset_name = dataset_name.lower() # e.g., "quickdraw"
        self.split = split
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.vector_transform = vector_transform
        self.raster_transform = raster_transform

        self.processed_sequences = [] # List to store loaded (dx,dy,p_state) sequences
        self.labels = []              # List to store labels
        
        # Load config to get category_map and potentially std_dev for denormalization if needed by user
        self.config_path = os.path.join(self.dataset_root_path, self.dataset_name, config_filename)
        self.config = {}
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            self.category_map = self.config.get("category_map", {})
            self.quickdraw_std_dev_for_denorm = self.config.get("quickdraw_std_dev") # For potential de-norm plotting
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
        # Data path now points to the directory containing preprocessed sequences for the split
        data_split_path = os.path.join(self.dataset_root_path, self.dataset_name, self.split)
        print(f"Loading PREPROCESSED {self.dataset_name} {self.split} data from: {data_split_path}")

        if not os.path.isdir(data_split_path):
            print(f"Error: Processed data directory not found: {data_split_path}")
            return

        categories_in_split = [f[:-4] for f in os.listdir(data_split_path) if f.endswith(".npy")]
        
        # If category_map is empty from config, try to build a temporary one (less ideal)
        if not self.category_map and categories_in_split:
            print("Building temporary category_map from directory listing as config map was empty/missing.")
            categories_in_split.sort() # Ensure consistent temp mapping
            for i, cat_name in enumerate(categories_in_split):
                self.category_map[cat_name] = i
        
        for category_name in categories_in_split:
            cat_label = self.category_map.get(category_name)
            if cat_label is None:
                print(f"Warning: Category '{category_name}' found in split dir but not in config's category_map. Skipping.")
                continue

            file_path = os.path.join(data_split_path, f"{category_name}.npy")
            try:
                # Each .npy file now contains a list of preprocessed sequence arrays
                # Needs allow_pickle=True because it's an array of arrays (objects)
                category_sequences = np.load(file_path, allow_pickle=True)
                for seq_array in category_sequences:
                    if isinstance(seq_array, np.ndarray) and seq_array.ndim == 2 and seq_array.shape[1] == 5:
                        self.processed_sequences.append(seq_array)
                        self.labels.append(cat_label)
                    else:
                        print(f"Warning: Skipping invalid item in {file_path}. Expected NumPy array (N,5). Got: {type(seq_array)}")

            except Exception as e:
                print(f"Error loading or processing preprocessed file {file_path}: {e}")
        
        if not self.processed_sequences:
            print(f"Warning: No processed sequences loaded for {self.dataset_name} {self.split}.")
        else:
            print(f"Loaded {len(self.processed_sequences)} processed sketches for {self.dataset_name} {self.split}.")


    def _vector_to_raster(self, processed_vector_sequence_np, image_size):
        # processed_vector_sequence_np is already (dx,dy,p0,p1,p2) and NORMALIZED
        # For plotting, we might want to denormalize if std_dev is known.
        # For internal consistency, rasterization will use the (normalized) deltas as is.
        # This means the resulting raster might appear "smaller" if std_dev was large.

        image = Image.new("L", (image_size, image_size), "white")
        draw = ImageDraw.Draw(image)
        temp_abs_points_segments = [] 
        current_abs_x, current_abs_y = 0.0, 0.0 

        for i in range(processed_vector_sequence_np.shape[0]):
            dx, dy, p_down, p_up, p_eos = processed_vector_sequence_np[i]
            next_abs_x, next_abs_y = current_abs_x + dx, current_abs_y + dy
            if i > 0 and p_down > 0.5: 
                temp_abs_points_segments.append(
                    ((current_abs_x, current_abs_y), (next_abs_x, next_abs_y))
                )
            current_abs_x, current_abs_y = next_abs_x, next_abs_y
            if p_eos > 0.5: break # Stop if end of sketch

        if not temp_abs_points_segments:
            image = image.convert("1"); return image

        all_x_coords = [p[0][0] for p in temp_abs_points_segments] + [p[1][0] for p in temp_abs_points_segments]
        all_y_coords = [p[0][1] for p in temp_abs_points_segments] + [p[1][1] for p in temp_abs_points_segments]
        if not all_x_coords or not all_y_coords: image = image.convert("1"); return image

        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        sketch_width = max_x - min_x
        sketch_height = max_y - min_y

        if sketch_width < 1e-6 and sketch_height < 1e-6:
            if temp_abs_points_segments: # Draw a point for single-point drawings
                 pt_x = (temp_abs_points_segments[0][0][0] - min_x) 
                 pt_y = (temp_abs_points_segments[0][0][1] - min_y) 
                 # Center this conceptual point
                 draw_x = image_size / 2 + pt_x # Simplified centering for a single point
                 draw_y = image_size / 2 + pt_y
                 draw.ellipse([(draw_x-1, draw_y-1), (draw_x+1, draw_y+1)], fill="black")
            image = image.convert("1"); return image

        padding_percent = 0.10
        canvas_draw_area_width = image_size * (1 - 2 * padding_percent)
        canvas_draw_area_height = image_size * (1 - 2 * padding_percent)
        scale_factor = 1.0
        if sketch_width > 1e-6: scale_factor = canvas_draw_area_width / sketch_width
        if sketch_height > 1e-6: scale_factor = min(scale_factor, canvas_draw_area_height / sketch_height)
        
        line_thickness = 2
        for p1_abs, p2_abs in temp_abs_points_segments:
            x1 = ((p1_abs[0] - min_x) * scale_factor) + (image_size - sketch_width * scale_factor) / 2
            y1 = ((p1_abs[1] - min_y) * scale_factor) + (image_size - sketch_height * scale_factor) / 2
            x2 = ((p2_abs[0] - min_x) * scale_factor) + (image_size - sketch_width * scale_factor) / 2
            y2 = ((p2_abs[1] - min_y) * scale_factor) + (image_size - sketch_height * scale_factor) / 2
            draw.line([(x1, y1), (x2, y2)], fill="black", width=line_thickness)
        del draw
        image = image.convert("1")
        return image

    def __len__(self):
        return len(self.processed_sequences)

    def __getitem__(self, idx):
        if not self.processed_sequences or idx >= len(self.processed_sequences):
            # Fallback for robustness, though should be handled by len check
            print(f"Warning: Index {idx} out of bounds or no data. Returning dummy item.")
            dummy_vec = torch.zeros((self.max_seq_len, 5), dtype=torch.float32)
            dummy_img = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            return {"vector_sketch": dummy_vec, "raster_image": dummy_img, 
                    "attention_mask": dummy_mask, "label": torch.tensor(-1, dtype=torch.long)}

        # Now loads an already processed (dx,dy,p_state) NORMALIZED sequence
        processed_normalized_sequence_np = self.processed_sequences[idx].copy() 
        
        seq_len = processed_normalized_sequence_np.shape[0]
        sketch_tensor = torch.tensor(processed_normalized_sequence_np, dtype=torch.float32)

        # Padding / Truncation
        if seq_len < self.max_seq_len:
            padding_tensor = torch.zeros((self.max_seq_len - seq_len, 5), dtype=torch.float32)
            sketch_tensor = torch.cat([sketch_tensor, padding_tensor], dim=0)
        elif seq_len > self.max_seq_len:
            sketch_tensor = sketch_tensor[:self.max_seq_len, :]
            if sketch_tensor[-1, 4] != 1: # Ensure last point is EOS if truncated
                sketch_tensor[-1, 0:2] = 0.0 
                sketch_tensor[-1, 2:4] = 0.0 
                sketch_tensor[-1, 4] = 1.0 
        
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        actual_len = min(seq_len, self.max_seq_len)
        attention_mask[:actual_len] = 1

        # Rasterization (uses the original length, normalized sequence)
        pil_image = self._vector_to_raster(processed_normalized_sequence_np, self.image_size)
        if self.raster_transform is None:
            self.raster_transform = transforms.ToTensor()
        raster_image_tensor = self.raster_transform(pil_image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "vector_sketch": sketch_tensor, "raster_image": raster_image_tensor,
            "attention_mask": attention_mask, "label": label
        }