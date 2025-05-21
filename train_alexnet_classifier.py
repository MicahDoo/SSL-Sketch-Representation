# train_alexnet_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import sys
import os
import time
import argparse
import json # For saving history
from datetime import datetime # For unique filenames
from tqdm import tqdm
from torch.amp import GradScaler, autocast # For PyTorch versions < 1.10, use torch.cuda.amp
import numpy as np
from PIL import Image  # for working with images
from torch.utils.data import Dataset, DataLoader  # for custom datasets


# --- Add project src to Python path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(SCRIPT_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
    print(f"Added '{SRC_PATH}' to sys.path")

try:
    from backbones.alexnet import AlexNetBackbone
    from models.alexnet_classifier import AlexNetClassifier
    from dataset import SketchDataset
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Make sure 'SRC_PATH' is correctly set and __init__.py files exist in your packages.")
    sys.exit(1)

# --- Configuration Function ---
def get_config():
    parser = argparse.ArgumentParser(description="Train AlexNet Classifier on QuickDraw Sketch Data")
    parser.add_argument('--num_classes', type=int, default=None, help="Number of sketch categories. If None, inferred from data.")
    parser.add_argument('--input_image_height', type=int, default=224, help="Height of input images.")
    parser.add_argument('--input_image_width', type=int, default=224, help="Width of input images.")
    parser.add_argument('--num_input_channels', type=int, default=1, help="Number of input image channels (1 for grayscale).")
    parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for training and evaluation.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--use_backbone_batch_norm', type=lambda x: (str(x).lower() == 'true'), default=True, help="Use BatchNorm in backbone.")
    parser.add_argument('--dropout_prob', type=float, default=0.5, help="Dropout probability in the classifier.")
    parser.add_argument('--data_dir_name', type=str, default="quickdraw_raster", help="Name of the data directory under processed_data.")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for DataLoader.")
    parser.add_argument('--save_model_name_prefix', type=str, default="alexnet_quickdraw", help="Prefix for saving the trained model filename.")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--amp', action='store_true', default=True, help='Enables Automatic Mixed Precision (AMP) training.')

    # Keep these params but they won't be used for caching anymore
    parser.add_argument('--use_cached_dataset', type=lambda x: (str(x).lower() == 'true'), 
                       default=False, help="Option kept for compatibility but not used")
    parser.add_argument('--preprocess_data', type=lambda x: (str(x).lower() == 'true'), 
                       default=False, help="Option kept for compatibility but not used")
    parser.add_argument('--preprocessed_data_dir', type=str, default="preprocessed_quickdraw", 
                       help="Option kept for compatibility but not used")
    parser.add_argument('--use_memory_mapped_files', type=lambda x: (str(x).lower() == 'true'), 
                       default=False, help="Option kept for compatibility but not used")
    parser.add_argument('--prefetch_factor', type=int, default=4, 
                       help="How many batches to prefetch per worker")

    args = parser.parse_args()
    return args

class MemoryMappedDataset(torch.utils.data.Dataset):
    """Kept for compatibility but won't be used in the optimized version"""
    def __init__(self, data_file, targets_file, transform=None):
        self.transform = transform
        # Memory-map the data files
        self.data = np.memmap(data_file, dtype=np.uint8, mode='r')
        # Reshape according to your image dimensions
        img_size = 224 * 224  # Adjust based on your config
        self.data = self.data.reshape(-1, img_size)
        self.targets = np.memmap(targets_file, dtype=np.int64, mode='r')
        
    def __getitem__(self, index):
        img = self.data[index].reshape(224, 224)  # Adjust shape based on your config
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.targets[index]
    
    def __len__(self):
        return len(self.targets)

# --- Data Loading Function ---
def load_data(config, project_root_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((config.input_image_height, config.input_image_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((config.input_image_height, config.input_image_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((config.input_image_height, config.input_image_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    }

    # Root of both raster and vector dirs
    sketch_data_root = os.path.join(project_root_dir, "processed_data")

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    num_classes_inferred = config.num_classes

    for split in ['train', 'val', 'test']:
        # Instantiate SketchDataset in raster mode
        ds = SketchDataset(
            root_dir=sketch_data_root,
            split=split,
            mode='raster',
            raster_transform=data_transforms[split]
        )
        image_datasets[split] = ds
        if len(ds) == 0:
            print(f"Warning: No samples found for {split} split.")
            dataloaders[split] = None
            dataset_sizes[split] = 0
            continue

        # Infer num classes on first non-empty split
        if split == 'train':
            inferred = len(ds.classes)
            if num_classes_inferred is None:
                num_classes_inferred = inferred
                print(f"Inferred NUM_CLASSES = {num_classes_inferred} from training data.")
            elif num_classes_inferred != inferred:
                print(f"Warning: Provided num_classes ({num_classes_inferred}) != actual ({inferred})."
                      f" Using actual.")
                num_classes_inferred = inferred

        dataloaders[split] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(split == 'train'),
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available() and not config.no_cuda,
            persistent_workers=(config.num_workers > 0),
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        )
        dataset_sizes[split] = len(ds)
        print(f"Loaded {split}: {dataset_sizes[split]} samples, {len(dataloaders[split])} batches")

    if num_classes_inferred is None or num_classes_inferred == 0:
        print("Critical Error: Number of classes could not be determined.")
        sys.exit(1)

    print(f"\nData loading summary: {dataset_sizes}")
    return dataloaders, num_classes_inferred

# --- Training Function ---
def train_model_loop(model, train_loader, criterion, optimizer, epoch_num, total_epochs, device, use_amp, scaler=None):
    model.train()
    running_loss = 0.0; correct_predictions = 0; total_samples = 0
    start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}/{total_epochs}", leave=False, unit="batch")

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if total_samples > 0:
            current_batch_loss = loss.item()
            current_batch_acc = (predicted == labels).sum().item() / labels.size(0)
            progress_bar.set_postfix(loss=f"{current_batch_loss:.4f}", acc=f"{current_batch_acc:.4f}")

    progress_bar.close()
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    epoch_time = time.time() - start_time
    print(f"Epoch [{epoch_num}/{total_epochs}] completed in {epoch_time:.2f}s - "
          f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

# --- Evaluation Function ---
def evaluate_model_loop(model, data_loader, criterion, device, use_amp, split_name="Validation"):
    if data_loader is None or len(data_loader) == 0:
        print(f"Skipping evaluation for {split_name}: Data loader is empty or not available.")
        return 0.0, 0.0
    model.eval()
    running_loss = 0.0; correct_predictions = 0; total_samples = 0
    progress_bar = tqdm(data_loader, desc=f"Evaluating {split_name}", leave=False, unit="batch")

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            if total_samples > 0:
                 progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(predicted == labels).sum().item() / labels.size(0):.4f}")

    progress_bar.close()
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"{split_name} Set: Average Loss: {avg_loss:.4f}, Accuracy: {correct_predictions}/{total_samples} ({accuracy:.4f})")
    return avg_loss, accuracy

# --- Main Execution ---
def main():
    config = get_config()
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Create a unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare directories for saved models and history
    model_save_dir = os.path.join(SCRIPT_DIR, "saved_model_weights")
    history_save_dir = os.path.join(SCRIPT_DIR, "training_history")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(history_save_dir, exist_ok=True)

    # Generate unique filenames
    unique_model_name = f"{config.save_model_name_prefix}_{timestamp}.pth"
    model_save_path = os.path.join(model_save_dir, unique_model_name)
    
    history_log_name = f"{config.save_model_name_prefix}_{timestamp}.json"
    history_log_path = os.path.join(history_save_dir, history_log_name)
    print(f"Training history will be saved to: {history_log_path}")
    print(f"Model will be saved to: {model_save_path}")


    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print("torch.backends.cudnn.benchmark is enabled.")
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            if hasattr(torch, 'set_float32_matmul_precision'):
                try:
                    torch.set_float32_matmul_precision('high')
                    print("torch.set_float32_matmul_precision('high') set for TF32.")
                except Exception as e:
                    print(f"Could not set float32 matmul precision: {e}")

    dataloaders, num_classes_actual = load_data(config, SCRIPT_DIR)
    final_num_classes = num_classes_actual if config.num_classes is None or config.num_classes != num_classes_actual else config.num_classes

    print(f"\nInstantiating AlexNetClassifier for {final_num_classes} classes...")
    model_to_train = AlexNetClassifier(
        num_classes=final_num_classes, input_image_height=config.input_image_height,
        input_image_width=config.input_image_width, num_input_channels=config.num_input_channels,
        backbone_use_batch_norm=config.use_backbone_batch_norm, dropout_prob=config.dropout_prob
    ).to(device)

    # Convert model to channel-last memory format for NHWC kernels
    model_to_train = model_to_train.to(memory_format=torch.channels_last)
    print("Model converted to channels_last format.")

    # Create example input for scripting
    example = torch.randn(1, config.num_input_channels, config.input_image_height, config.input_image_width).to(device)
    model_to_train.eval()
    scripted_model = torch.jit.trace(model_to_train, example)
    print("Model scripted via TorchScript")

    # Optional: compile with channel-last support
    if hasattr(torch, 'compile') and device.type == 'cuda':
        compiled_model = torch.compile(scripted_model, mode="reduce-overhead")
    else:
        compiled_model = scripted_model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(compiled_model.parameters(), lr=config.learning_rate)
    print("Loss function and optimizer defined.")

    # With:
    optimizer = optim.SGD(
        compiled_model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=0.0001
    )
    # And add a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    scaler = None
    use_amp_for_training = device.type == 'cuda' and config.amp
    if use_amp_for_training:
        scaler = torch.amp.GradScaler(enabled=use_amp_for_training)
        print("Automatic Mixed Precision (AMP) training enabled with GradScaler.")
    else:
        if device.type != 'cuda': print("AMP training disabled (CUDA not available).")
        elif not config.amp: print("AMP training disabled (--amp flag not set).")

    training_history = [] # Initialize list to store history

    if dataloaders.get('train'):
        print("\nStarting training...")
        for epoch in range(1, config.num_epochs + 1): # Epochs 1-indexed for logging
            train_loss, train_acc = train_model_loop(
                compiled_model, dataloaders['train'], criterion, optimizer, 
                epoch, config.num_epochs, device, use_amp_for_training, scaler
            )
            
            val_loss, val_acc = 0.0, 0.0
            if dataloaders.get('val'):
                val_loss, val_acc = evaluate_model_loop(
                    compiled_model, dataloaders['val'], criterion, device, 
                    use_amp_for_training, # Use same AMP setting for val consistency
                    split_name="Validation"
                )

            scheduler.step()
            
            epoch_history = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            training_history.append(epoch_history)
            
            # Save history to JSON file after each epoch
            try:
                with open(history_log_path, 'w') as f:
                    json.dump(training_history, f, indent=4)
                # print(f"History updated and saved to {history_log_path}")
            except Exception as e:
                print(f"Error saving training history after epoch {epoch}: {e}")
    else:
        print("Skipping training: Training data loader not available.")

    if dataloaders.get('test'):
        print("\nEvaluating on Test Set...")
        # Test set evaluation can also be added to history if desired,
        # or logged separately. For now, just printing.
        evaluate_model_loop(compiled_model, dataloaders['test'], criterion, device, 
                            use_amp_for_training, # Use same AMP for consistency
                            split_name="Test")

    try:
        if hasattr(compiled_model, '_orig_mod'):
            torch.save(compiled_model._orig_mod.state_dict(), model_save_path)
            print(f"\nOriginal model state_dict (from compiled model) saved to: {model_save_path}")
        else:
            torch.save(model_to_train.state_dict(), model_save_path)
            print(f"\nTrained model state_dict saved to: {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print(f"\nFinal training history saved to: {history_log_path}")

if __name__ == "__main__":
    # torch.manual_seed(RANDOM_SEED)
    # np.random.seed(RANDOM_SEED)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(RANDOM_SEED)
        # Optional: for full determinism, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    main()