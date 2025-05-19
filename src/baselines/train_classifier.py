# train_classifier.py
import sys
import os

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
import shutil 

# For Automatic Mixed Precision (AMP)
from torch.amp import autocast 
from torch.cuda.amp import GradScaler # Keep this for fallback or if torch.amp.GradScaler not available

try:
    from src.models.raster_encoder import ResNet50SketchEncoderBase
    from data.dataset import SketchDataset 
    print("DEBUG: Successfully imported ResNet50SketchEncoderBase and SketchDataset.")
except ImportError as e:
    print(f"ERROR: Error importing modules: {e}")
    print("DEBUG: Please ensure 'src/models/raster_encoder.py' and 'data/dataset.py' exist.")
    ResNet50SketchEncoderBase = None
    SketchDataset = None

# --- Configuration ---
DEFAULT_SKETCH_IMG_SIZE = 224
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "processed_data") 
DEFAULT_DATASET_NAME = "quickdraw"
DEFAULT_CONFIG_FILENAME = "quickdraw_config.json"
DEFAULT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

class SketchClassifier(nn.Module):
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

def save_checkpoint(epoch, model, optimizer, scheduler, history, best_val_acc, checkpoint_path, is_best=False):
    checkpoint = {
        'epoch': epoch + 1, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history
    }
    if scheduler and hasattr(scheduler, 'state_dict'): # Check if scheduler has state_dict
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    if is_best:
        best_path = os.path.join(os.path.dirname(checkpoint_path), f"{os.path.basename(checkpoint_path).replace('_latest_checkpoint','_best_model')}")
        shutil.copyfile(checkpoint_path, best_path)
        print(f"Best model checkpoint saved to {best_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        return 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}, 0.0
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    history = checkpoint.get('history', {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}) 
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    if scheduler and hasattr(scheduler, 'state_dict') and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded.")
    else:
        if scheduler and hasattr(scheduler, 'state_dict'):
            print("Warning: Scheduler state not found in checkpoint or scheduler has no state_dict.")
    
    print(f"Resuming training from epoch {start_epoch}. Best val_acc so far: {best_val_acc:.4f}")
    return start_epoch, history, best_val_acc


def train_and_validate(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, 
                       num_epochs, device, experiment_name="Classifier", use_amp=False,
                       checkpoint_dir="checkpoints", save_every=1, start_epoch=0, 
                       initial_history=None, initial_best_val_acc=0.0,
                       profile_steps=False): 
    
    history = initial_history if initial_history is not None else {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = initial_best_val_acc
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_latest_checkpoint.pth")
    
    scaler = None
    amp_on_cuda = use_amp and device.type == 'cuda'

    if amp_on_cuda:
        try:
            # Try the torch.amp.GradScaler (newer PyTorch versions)
            scaler = torch.amp.GradScaler(enabled=amp_on_cuda) # Removed device_type
            print(f"DEBUG: Using torch.amp.GradScaler for {experiment_name} on CUDA.")
        except AttributeError: 
            # Fallback to torch.cuda.amp.GradScaler (older PyTorch versions)
            scaler = GradScaler(enabled=amp_on_cuda)
            print(f"DEBUG: Using torch.cuda.amp.GradScaler for {experiment_name} on CUDA.")
            
    elif use_amp and device.type == 'cpu':
        print(f"DEBUG: AMP with torch.amp.autocast('{device.type}') will be used for {experiment_name} on CPU (typically bfloat16 if supported).")
    
    print(f"\n--- Starting Training: {experiment_name} from epoch {start_epoch} for {num_epochs} total epochs on {device} ---")

    prof = None # Define prof outside the conditional block
    if profile_steps and start_epoch == 0 : 
        print(f"DEBUG: Profiler enabled.")
        activities_to_profile = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities_to_profile.append(torch.profiler.ProfilerActivity.CUDA)
        trace_dir = os.path.join(checkpoint_dir, f'{experiment_name}_profile_trace')
        os.makedirs(trace_dir, exist_ok=True)
        print(f"DEBUG: Profiler traces will be saved to: {trace_dir}")
        prof = torch.profiler.profile(
            activities=activities_to_profile,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir), 
            record_shapes=True, profile_memory=True, with_stack=True
        )
        # prof.start() # Not needed with schedule

    # Training loop
    for epoch in range(start_epoch, num_epochs): 
        current_epoch_display = epoch + 1 
        start_time_epoch = time.time()
        model.train()
        running_train_loss = 0.0; correct_train_preds = 0; total_train_samples = 0
        
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {current_epoch_display}/{num_epochs} [Train]", leave=False)
        
        # Conditional start of profiler context for the first few batches of the first epoch
        if prof and epoch == start_epoch:
            print("DEBUG: Profiler context starting for initial batches...")
            with prof:
                for batch_idx, data_batch in enumerate(batch_iterator):
                    if batch_idx >= (1 + 3) : # PROF_WARMUP + PROF_ACTIVE * PROF_REPEAT
                        # After profiling enough steps, break out of profiler context for this epoch
                        # The main loop will continue, but outside the profiler context
                        break 
                    
                    inputs = data_batch['raster_image'].to(device)
                    targets = data_batch['label'].to(device)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp): # autocast for forward pass
                        outputs = model(inputs); loss = loss_fn(outputs, targets)
                    if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    else: loss.backward(); optimizer.step()
                    
                    running_train_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct_train_preds += torch.sum(preds == targets.data).item()
                    total_train_samples += targets.size(0)
                    prof.step() # Important: prof.step() inside the profiled loop
            print("DEBUG: Profiler context for initial batches ended.")
            # Reset prof to None or profile_steps to False if you only want to profile once per run
            # profile_steps = False 
            
        # Continue with regular loop if not profiling or after profiling window
        if not (prof and epoch == start_epoch): # If not in the special profiling block above
            for batch_idx, data_batch in enumerate(batch_iterator): # Use the same iterator
                # Skip if already processed by profiler (this logic needs care)
                if prof and epoch == start_epoch and batch_idx < (1+3): continue

                inputs = data_batch['raster_image'].to(device)
                targets = data_batch['label'].to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs); loss = loss_fn(outputs, targets)
                if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else: loss.backward(); optimizer.step()
                running_train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train_preds += torch.sum(preds == targets.data).item()
                total_train_samples += targets.size(0)

        epoch_train_loss = running_train_loss / total_train_samples if total_train_samples > 0 else 0
        epoch_train_acc = correct_train_preds / total_train_samples if total_train_samples > 0 else 0
        history['train_loss'].append(epoch_train_loss); history['train_acc'].append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0; correct_val_preds = 0; total_val_samples = 0
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                for batch_idx, data_batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {current_epoch_display}/{num_epochs} [Val]", leave=False)):
                    inputs = data_batch['raster_image'].to(device)
                    targets = data_batch['label'].to(device)
                    outputs = model(inputs); loss = loss_fn(outputs, targets)
                    running_val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct_val_preds += torch.sum(preds == targets.data).item()
                    total_val_samples += targets.size(0)
        epoch_val_loss = running_val_loss / total_val_samples if total_val_samples > 0 else 0
        epoch_val_acc = correct_val_preds / total_val_samples if total_val_samples > 0 else 0
        history['val_loss'].append(epoch_val_loss); history['val_acc'].append(epoch_val_acc)
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(epoch_val_loss)
            else: scheduler.step()
        
        epoch_duration = time.time() - start_time_epoch
        print(f"Epoch {current_epoch_display}/{num_epochs} [{experiment_name}] - Dur: {epoch_duration:.1f}s")
        print(f"  TrL: {epoch_train_loss:.4f}, TrA: {epoch_train_acc:.4f} | VaL: {epoch_val_loss:.4f}, VaA: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        is_best = epoch_val_acc > best_val_acc
        if is_best: best_val_acc = epoch_val_acc
        if (epoch + 1) % save_every == 0 or is_best or (epoch + 1) == num_epochs : 
            save_checkpoint(epoch, model, optimizer, scheduler, history, best_val_acc, latest_checkpoint_path, is_best=is_best)
            
    # After the training loop, for scheduled profiler
    if prof and profile_steps and epoch == start_epoch : # Only print if it ran
        print("\n--- Profiler Results (Top 15 by self CUDA time or CPU time if CUDA unavailable) ---")
        sort_by_key = "self_cuda_time_total" if device.type == 'cuda' else "self_cpu_time_total"
        try: print(prof.key_averages().table(sort_by=sort_by_key, row_limit=15))
        except Exception as e: print(f"Could not print profiler table: {e}")
        print(f"Profiler trace saved to directory: {os.path.join(checkpoint_dir, f'{experiment_name}_profile_trace')}")

    print(f"Training for {experiment_name} complete! Best Val Acc: {best_val_acc:.4f}")
    return history

def main(args):
    # ... (main function setup as before) ...
    # Key part is passing args.profile to train_and_validate
    if SketchDataset is None or ResNet50SketchEncoderBase is None: print("Required modules not imported. Exiting."); return
    if args.use_gpu and torch.cuda.is_available():
        if args.gpu_id is not None:
            if args.gpu_id < torch.cuda.device_count(): device_str = f"cuda:{args.gpu_id}"
            else: print(f"Warning: GPU ID {args.gpu_id} invalid. Using default cuda:0."); device_str = "cuda:0"
        else: device_str = "cuda"
        device = torch.device(device_str)
    else:
        if args.use_gpu and not torch.cuda.is_available(): print("Warning: --use_gpu specified, but CUDA not available. Using CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if not os.path.isabs(args.dataset_root): dataset_root_abs = os.path.join(PROJECT_ROOT, args.dataset_root)
    else: dataset_root_abs = args.dataset_root
    vector_data_root = os.path.join(dataset_root_abs, f"{args.dataset_name}_vector")
    main_config_path = os.path.join(vector_data_root, args.config_filename)
    num_classes = None
    if os.path.exists(main_config_path):
        with open(main_config_path, 'r') as f: config = json.load(f)
        num_classes = len(config.get("category_map", {}))
        print(f"Loaded config from {main_config_path}. Number of classes: {num_classes}")
    else:
        if args.num_classes: num_classes = args.num_classes
        else: print("Error: num_classes not determined."); return
    if not num_classes or num_classes == 0: print("Error: Number of classes is 0 or not determined."); return
    input_channels_for_model = 1; raster_image_transform = transforms.ToTensor()
    if args.use_pretrained:
        input_channels_for_model = 3
        raster_image_transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.ndim == 3 and x.shape[0]==1 else x), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = SketchDataset(dataset_path=dataset_root_abs, dataset_name=args.dataset_name, split=args.train_split, image_size=args.img_size, max_seq_len=args.max_seq_len, config_filename=args.config_filename, raster_transform=raster_image_transform)
    val_dataset = SketchDataset(dataset_path=dataset_root_abs, dataset_name=args.dataset_name, split=args.val_split, image_size=args.img_size, max_seq_len=args.max_seq_len, config_filename=args.config_filename, raster_transform=raster_image_transform)
    if len(train_dataset) == 0 or len(val_dataset) == 0: print("Training or validation dataset is empty."); return
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"DataLoaders created. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")
    model = SketchClassifier(num_classes=num_classes, input_channels=input_channels_for_model, use_pretrained_backbone=args.use_pretrained, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = args.lr if not args.use_pretrained else args.lr * 0.1 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, factor=0.1) 
    start_epoch = 0; initial_history = None; initial_best_val_acc = 0.0
    experiment_name = f"ResNetClassify_Prtnd_{args.use_pretrained}_AMP_{args.use_amp and device.type=='cuda'}"
    checkpoint_dir_abs = os.path.join(PROJECT_ROOT, args.checkpoint_dir) if not os.path.isabs(args.checkpoint_dir) else args.checkpoint_dir
    latest_checkpoint_file = os.path.join(checkpoint_dir_abs, f"{experiment_name}_latest_checkpoint.pth")
    if args.resume:
        if os.path.exists(latest_checkpoint_file):
            start_epoch, initial_history, initial_best_val_acc = load_checkpoint(model, optimizer, scheduler, latest_checkpoint_file, device)
        else: print(f"Resume requested, but no checkpoint found at {latest_checkpoint_file}. Starting fresh.")
    else: print("Starting training from scratch (no resume).")
    history = train_and_validate(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler,
                                 num_epochs=args.epochs, device=device, experiment_name=experiment_name,
                                 use_amp=args.use_amp, checkpoint_dir=checkpoint_dir_abs, 
                                 save_every=args.save_every, start_epoch=start_epoch,
                                 initial_history=initial_history, initial_best_val_acc=initial_best_val_acc,
                                 profile_steps=args.profile) # Pass profile flag
    if history:
        plot_save_path = os.path.join(checkpoint_dir_abs, f"{experiment_name}_final_training_history.png") 
        plot_training_history(history, experiment_name, save_path=plot_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ResNet Sketch Classifier.")
    parser.add_argument('--dataset_root', type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--dataset_name', type=str, default=DEFAULT_DATASET_NAME)
    # ... (other args) ...
    parser.add_argument('--config_filename', type=str, default=DEFAULT_CONFIG_FILENAME)
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val')
    parser.add_argument('--img_size', type=int, default=DEFAULT_SKETCH_IMG_SIZE)
    parser.add_argument('--max_seq_len', type=int, default=70)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size. Increase if VRAM allows.")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (AMP).') 
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler for a few initial steps of the first epoch.')

    args = parser.parse_args()
    print("\n--- Parsed Arguments ---"); print(vars(args)); print("------------------------\n")
    main(args)
