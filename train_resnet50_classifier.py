# train_resnet50_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import sys
import os
import time
import argparse
import json  # For saving history
from datetime import datetime  # For unique filenames
from tqdm import tqdm
from torch.amp import GradScaler, autocast  # For AMP
import numpy as np
from PIL import Image  # for working with images
from torch.utils.data import Dataset, DataLoader  # for custom datasets

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42

# --- Add project src to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(SCRIPT_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    print(f"Added '{SRC_PATH}' to sys.path")

try:
    from backbones.resnet50 import ResNet50Backbone
    from models.resnet50 import ResNet50Classifier
    from dataset import SketchDataset
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Make sure 'SRC_PATH' is correctly set and __init__.py files exist in your packages.")
    sys.exit(1)

# --- Configuration Function ---
def get_config():
    parser = argparse.ArgumentParser(description="Train ResNet50 Classifier on QuickDraw Sketch Data")
    parser.add_argument('--num_classes', type=int, default=None, help="Number of sketch categories. If None, inferred from data.")
    parser.add_argument('--input_image_height', type=int, default=224, help="Height of input images.")
    parser.add_argument('--input_image_width', type=int, default=224, help="Width of input images.")
    parser.add_argument('--num_input_channels', type=int, default=1, help="Number of input image channels (1 for grayscale).")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for training and evaluation.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--backbone_pretrained', type=lambda x: (str(x).lower() == 'true'),
                        default=True, help="Load ImageNet-pretrained weights for backbone.")
    parser.add_argument('--dropout_prob', type=float, default=0.5, help="Dropout probability in the classifier.")
    parser.add_argument('--data_dir_name', type=str, default="quickdraw_raster",
                        help="Name of the data directory under processed_data.")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for DataLoader.")
    parser.add_argument('--save_model_name_prefix', type=str,
                        default="resnet50_quickdraw", help="Prefix for saving the trained model filename.")
    parser.add_argument('--history_log_name', type=str, default="training_history.json",
                        help="Filename for saving the training history log.")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--amp', action='store_true', default=True, help='Enables Automatic Mixed Precision (AMP) training.')

    # Legacy args (kept for compatibility)
    parser.add_argument('--use_cached_dataset', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help="Not used")
    parser.add_argument('--preprocess_data', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help="Not used")
    parser.add_argument('--preprocessed_data_dir', type=str, default="preprocessed_quickdraw",
                        help="Not used")
    parser.add_argument('--use_memory_mapped_files', type=lambda x: (str(x).lower() == 'true'),
                        default=False, help="Not used")
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help="How many batches to prefetch per worker (if num_workers>0)")

    return parser.parse_args()

# --- Data Loading Function (same as AlexNet version) ---
def load_data(config, project_root_dir):
    data_transforms = {
        split: transforms.Compose([
            transforms.Grayscale(num_output_channels=config.num_input_channels),
            transforms.Resize((config.input_image_height, config.input_image_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * config.num_input_channels, (0.5,) * config.num_input_channels)
        ]) for split in ['train', 'val', 'test']
    }

    sketch_data_root = os.path.join(project_root_dir, "processed_data")
    image_datasets, dataloaders, dataset_sizes = {}, {}, {}
    num_classes_inferred = config.num_classes

    for split in ['train', 'val', 'test']:
        ds = SketchDataset(
            root_dir=sketch_data_root,
            split=split,
            mode='raster',
            raster_transform=data_transforms[split]
        )
        image_datasets[split] = ds
        if len(ds) == 0:
            print(f"Warning: No samples found for {split} split.")
            dataloaders[split], dataset_sizes[split] = None, 0
            continue

        if split == 'train':
            inferred = len(ds.classes)
            if num_classes_inferred is None:
                num_classes_inferred = inferred
                print(f"Inferred NUM_CLASSES = {num_classes_inferred} from training data.")
            elif num_classes_inferred != inferred:
                print(f"Warning: Provided num_classes ({num_classes_inferred}) != actual ({inferred}). Using actual.")
                num_classes_inferred = inferred

        dataloaders[split] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(split == 'train'),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available() and not config.no_cuda,
            persistent_workers=(config.num_workers > 0),
            prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
        )
        dataset_sizes[split] = len(ds)
        print(f"Loaded {split}: {dataset_sizes[split]} samples, {len(dataloaders[split]) if dataloaders[split] else 0} batches")

    if not num_classes_inferred:
        print("Critical Error: Number of classes could not be determined.")
        sys.exit(1)

    print(f"\nData loading summary: {dataset_sizes}")
    return dataloaders, num_classes_inferred

# --- Training & Evaluation Loops (unchanged) ---
def train_model_loop(model, train_loader, criterion, optimizer, epoch_num, total_epochs, device, use_amp, scaler=None):
    model.train()
    running_loss = correct = total = 0
    start = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{total_epochs}", leave=False, unit="batch")
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(preds==labels).float().mean():.4f}")
    pbar.close()
    epoch_time = time.time() - start
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch [{epoch_num}/{total_epochs}] done in {epoch_time:.2f}s - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def evaluate_model_loop(model, loader, criterion, device, use_amp, split_name="Validation"):
    if not loader:
        print(f"Skipping {split_name}: no data.")
        return 0.0, 0.0
    model.eval()
    running_loss = correct = total = 0
    pbar = tqdm(loader, desc=f"Evaluating {split_name}", leave=False, unit="batch")
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(preds==labels).float().mean():.4f}")
    pbar.close()
    avg_loss, acc = running_loss/total, correct/total
    print(f"{split_name} - Loss: {avg_loss:.4f}, Acc: {correct}/{total} ({acc:.4f})")
    return avg_loss, acc

# --- Main Execution ---
def main():
    config = get_config()
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(SCRIPT_DIR, "saved_model_weights")
    history_dir = os.path.join(SCRIPT_DIR, "training_history")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    model_name = f"{config.save_model_name_prefix}_{timestamp}.pth"
    model_path = os.path.join(model_dir, model_name)
    history_path = os.path.join(history_dir, config.history_log_name)
    print(f"Model will be saved to: {model_path}")
    print(f"History will be saved to: {history_path}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print("cudnn.benchmark enabled.")
        if torch.cuda.get_device_capability()[0] >= 8 and hasattr(torch, 'set_float32_matmul_precision'):
            try:
                torch.set_float32_matmul_precision('high')
                print("TF32 precision enabled.")
            except Exception as e:
                print(f"Could not set TF32: {e}")

    dataloaders, num_classes_actual = load_data(config, SCRIPT_DIR)
    final_classes = num_classes_actual if config.num_classes is None or config.num_classes != num_classes_actual else config.num_classes

    print(f"\nInstantiating ResNet50Classifier for {final_classes} classes...")
    model = ResNet50Classifier(
        num_classes=final_classes,
        input_image_height=config.input_image_height,
        input_image_width=config.input_image_width,
        num_input_channels=config.num_input_channels,
        backbone_pretrained=config.backbone_pretrained,
        dropout_prob=config.dropout_prob
    ).to(device)

    model = model.to(memory_format=torch.channels_last)
    print("Converted model to channels_last format.")

    example = torch.randn(1, config.num_input_channels, config.input_image_height, config.input_image_width).to(device)
    model.eval()
    scripted = torch.jit.trace(model, example)
    print("Model traced with TorchScript.")

    compiled = torch.compile(scripted, mode="reduce-overhead") if hasattr(torch, 'compile') and device.type=='cuda' else scripted

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(compiled.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print("Loss, optimizer, and scheduler defined.")

    scaler = GradScaler(enabled=(device.type=='cuda' and config.amp))
    if scaler.is_enabled():
        print("AMP training enabled.")
    else:
        print("AMP disabled.")

    history = []
    if dataloaders.get('train'):
        for epoch in range(1, config.num_epochs+1):
            train_loss, train_acc = train_model_loop(
                compiled, dataloaders['train'], criterion, optimizer,
                epoch, config.num_epochs, device, scaler.is_enabled(), scaler
            )
            val_loss, val_acc = evaluate_model_loop(
                compiled, dataloaders.get('val'), criterion, device,
                scaler.is_enabled(), split_name="Validation"
            )
            scheduler.step()

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
    else:
        print("No training data; skipping training.")

    if dataloaders.get('test'):
        print("\n--- Testing ---")
        evaluate_model_loop(compiled, dataloaders['test'], criterion, device, scaler.is_enabled(), split_name="Test")

    try:
        state_dict = compiled._orig_mod.state_dict() if hasattr(compiled, '_orig_mod') else model.state_dict()
        torch.save(state_dict, model_path)
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print(f"Training complete. History at: {history_path}")

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    main()
