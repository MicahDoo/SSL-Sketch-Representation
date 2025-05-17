# train_classifier.py
import sys
import os

# --- Path Setup ---
# Get the directory containing this script (src/baselines/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (which should be two levels up from src/baselines/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"DEBUG: Added {PROJECT_ROOT} to sys.path")
else:
    print(f"DEBUG: {PROJECT_ROOT} is already in sys.path")

print(f"DEBUG: Current sys.path[0]: {sys.path[0] if sys.path else 'EMPTY'}")
print(f"DEBUG: SCRIPT_DIR: {SCRIPT_DIR}")
print(f"DEBUG: PROJECT_ROOT: {PROJECT_ROOT}")
# --- End Path Setup ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms 
import argparse
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Attempt to import from project structure
try:
    from src.models.raster_encoder import ResNet50SketchEncoderBase
    from data.dataset import SketchDataset # Your SketchDataset
    print("DEBUG: Successfully imported ResNet50SketchEncoderBase and SketchDataset.")
except ImportError as e:
    print(f"ERROR: Error importing modules: {e}")
    print("DEBUG: Please ensure 'src/models/raster_encoder.py' and 'data/dataset.py' exist.")
    print(f"DEBUG: Project root (added to path): {PROJECT_ROOT}")
    print(f"DEBUG: Effective sys.path for imports starts with: {sys.path[0] if sys.path else 'EMPTY'}")
    print("DEBUG: Check that 'src' and 'data' are direct subdirectories of the project root,")
    print("       and that they (and their submodules like 'models') have __init__.py files.")
    ResNet50SketchEncoderBase = None
    SketchDataset = None

# --- Configuration ---
DEFAULT_SKETCH_IMG_SIZE = 224
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "processed_data") 
DEFAULT_DATASET_NAME = "quickdraw"
DEFAULT_CONFIG_FILENAME = "quickdraw_config.json"

class SketchClassifier(nn.Module):
    """
    A classifier model that uses the ResNet50SketchEncoderBase as a backbone
    and adds a classification head.
    """
    def __init__(self, num_classes, input_channels=1, use_pretrained_backbone=False, freeze_backbone=False):
        super(SketchClassifier, self).__init__()
        self.backbone = ResNet50SketchEncoderBase(
            input_channels=input_channels,
            use_pretrained=use_pretrained_backbone,
            freeze_pretrained=freeze_backbone
        )
        self.classifier_head = nn.Linear(self.backbone.output_feature_dim, num_classes)
        print(f"SketchClassifier initialized for {num_classes} classes.")
        print(f"  Backbone output features: {self.backbone.output_feature_dim}")
        if use_pretrained_backbone:
            print(f"  Backbone uses pre-trained weights (ImageNet). Frozen: {freeze_backbone}")
        else:
            print("  Backbone trained from scratch.")


    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits

def train_and_validate(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, 
                       num_epochs, device, experiment_name="Classifier"):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"\n--- Starting Training: {experiment_name} for {num_epochs} epochs on {device} ---")

    for epoch in range(num_epochs):
        start_time_epoch = time.time()
        
        model.train()
        running_train_loss = 0.0
        correct_train_preds = 0
        total_train_samples = 0

        for batch_idx, data_batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)):
            inputs = data_batch['raster_image'].to(device)
            targets = data_batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train_preds += torch.sum(preds == targets.data).item()
            total_train_samples += targets.size(0)
            
        epoch_train_loss = running_train_loss / total_train_samples if total_train_samples > 0 else 0
        epoch_train_acc = correct_train_preds / total_train_samples if total_train_samples > 0 else 0
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)):
                inputs = data_batch['raster_image'].to(device)
                targets = data_batch['label'].to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val_preds += torch.sum(preds == targets.data).item()
                total_val_samples += targets.size(0)

        epoch_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0
        epoch_val_acc = correct_val_preds / total_val_samples if total_val_samples > 0 else 0
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else: 
                scheduler.step()
        
        epoch_duration = time.time() - start_time_epoch
        print(f"Epoch {epoch+1}/{num_epochs} [{experiment_name}] - Dur: {epoch_duration:.1f}s")
        print(f"  TrL: {epoch_train_loss:.4f}, TrA: {epoch_train_acc:.4f} | VaL: {epoch_val_loss:.4f}, VaA: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # torch.save(model.state_dict(), f"{experiment_name.replace(' ','_')}_best_model.pth")
            # print(f"    New best validation accuracy: {best_val_acc:.4f}. Model checkpoint saved.")
            
    print(f"Training for {experiment_name} complete! Best Val Acc: {best_val_acc:.4f}")
    return history

def plot_training_history(history, experiment_name="Experiment", save_path=None):
    if not history: print(f"No training history to plot for {experiment_name}."); return
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Training History: {experiment_name}", fontsize=16)
    axs[0].plot(history['train_acc'], label='Train Acc', marker='.'); axs[0].plot(history['val_acc'], label='Val Acc', marker='.')
    axs[0].set_title('Model Accuracy'); axs[0].set_ylabel('Accuracy'); axs[0].set_xlabel('Epoch'); axs[0].legend(loc='lower right'); axs[0].grid(True)
    axs[1].plot(history['train_loss'], label='Train Loss', marker='.'); axs[1].plot(history['val_loss'], label='Val Loss', marker='.')
    axs[1].set_title('Model Loss'); axs[1].set_ylabel('Loss'); axs[1].set_xlabel('Epoch'); axs[1].legend(loc='upper right'); axs[1].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    plt.show()


def main(args):
    if SketchDataset is None or ResNet50SketchEncoderBase is None:
        print("Required modules (SketchDataset or ResNet50SketchEncoderBase) not imported. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    if not os.path.isabs(args.dataset_root):
        dataset_root_abs = os.path.join(PROJECT_ROOT, args.dataset_root)
        print(f"DEBUG: Relative dataset_root '{args.dataset_root}' resolved to absolute path: '{dataset_root_abs}'")
    else:
        dataset_root_abs = args.dataset_root
        print(f"DEBUG: Using absolute dataset_root: '{dataset_root_abs}'")

    vector_data_root = os.path.join(dataset_root_abs, f"{args.dataset_name}_vector")
    main_config_path = os.path.join(vector_data_root, args.config_filename)
    
    num_classes = None
    if os.path.exists(main_config_path):
        with open(main_config_path, 'r') as f:
            config = json.load(f)
            num_classes = len(config.get("category_map", {}))
            print(f"Loaded config from {main_config_path}. Number of classes: {num_classes}")
    else:
        print(f"Warning: Main config file not found at {main_config_path}. Cannot determine num_classes automatically.")
        if args.num_classes:
            num_classes = args.num_classes
            print(f"Using num_classes from arguments: {num_classes}")
        else:
            print("Error: num_classes not determined. Please provide --num_classes or ensure config file exists.")
            return
            
    if not num_classes or num_classes == 0: # Added check for num_classes being None or 0
        print("Error: Number of classes is 0 or not determined. Check config or --num_classes argument.")
        return

    input_channels = 1
    raster_image_transform = None 
    
    if args.use_pretrained:
        input_channels = 3 
        print("Using pretrained backbone. Expecting 3-channel normalized input to model.")
    else:
        print("Training backbone from scratch. Expecting 1-channel input to model.")

    print(f"Initializing SketchDataset for training (split: {args.train_split})...")
    train_dataset = SketchDataset(
        dataset_path=dataset_root_abs, 
        dataset_name=args.dataset_name,
        split=args.train_split,
        image_size=args.img_size,
        max_seq_len=args.max_seq_len, 
        config_filename=args.config_filename,
        raster_transform=raster_image_transform 
    )
    print(f"Initializing SketchDataset for validation (split: {args.val_split})...")
    val_dataset = SketchDataset(
        dataset_path=dataset_root_abs, 
        dataset_name=args.dataset_name,
        split=args.val_split,
        image_size=args.img_size,
        max_seq_len=args.max_seq_len,
        config_filename=args.config_filename,
        raster_transform=raster_image_transform 
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Training or validation dataset is empty. Please check data paths and preprocessing.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"DataLoaders created. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    model = SketchClassifier(
        num_classes=num_classes,
        input_channels=input_channels,
        use_pretrained_backbone=args.use_pretrained,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = args.lr if not args.use_pretrained else args.lr * 0.1 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # Removed verbose=True for compatibility with older PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, factor=0.1) 

    experiment_name = f"ResNetClassifier_Pretrained_{args.use_pretrained}"
    history = train_and_validate(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                                 num_epochs=args.epochs, device=device, experiment_name=experiment_name)
    
    if history:
        plot_save_path = f"{experiment_name}_training_history.png" 
        plot_training_history(history, experiment_name, save_path=plot_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ResNet Sketch Classifier on QuickDraw (or similar).")
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT, 
                        help=f"Root directory of the processed dataset (default: relative to project root as '{os.path.basename(DEFAULT_DATASET_ROOT)}').")
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME, help='Name of the dataset (e.g., quickdraw).')
    parser.add_argument('--config_filename', type=str, default=DEFAULT_CONFIG_FILENAME, help='Name of the config file in the dataset_name_vector directory.')
    parser.add_argument('--train_split', type=str, default='train', help='Name of the training split.')
    parser.add_argument('--val_split', type=str, default='val', help='Name of the validation split.')
    parser.add_argument('--img_size', type=int, default=DEFAULT_SKETCH_IMG_SIZE, help='Image size for raster sketches.')
    parser.add_argument('--max_seq_len', type=int, default=70, help='Max sequence length for vector part (passed to SketchDataset).')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes (if not determinable from config).')
    parser.add_argument('--use_pretrained', action='store_true', help='Use ImageNet pre-trained weights for ResNet backbone.')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze weights of the ResNet backbone (only classifier head trains).')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer.')
    parser.add_argument('--lr_patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    args = parser.parse_args()
    
    print("\n--- Parsed Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("------------------------\n")

    main(args)
