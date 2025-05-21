import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image # Pillow is a dependency of torchvision
import time
import copy

# Configuration
BASE_DATA_DIR = 'processed_data/quickdraw_raster/'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'val')
# TEST_DIR = os.path.join(BASE_DATA_DIR, 'test') # Path for test set, not used in this training script

MODEL_SAVE_PATH = 'quickdraw_resnet50_presplit.pth'
NUM_EPOCHS = 25  # Number of epochs to train for
BATCH_SIZE = 32
LEARNING_RATE = 0.001
RANDOM_SEED = 42 # For reproducibility of any stochasticity within PyTorch

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    """
    Main function to orchestrate the dataset loading, model training, and evaluation.
    """

    # --- 1. Check if data directories exist ---
    if not os.path.isdir(TRAIN_DIR):
        print(f"Error: Training data directory '{TRAIN_DIR}' not found.")
        print("Please ensure your 'train' subfolder with class directories exists within 'processed_data/quickdraw_raster/'.")
        return
    if not os.path.isdir(VAL_DIR):
        print(f"Error: Validation data directory '{VAL_DIR}' not found.")
        print("Please ensure your 'val' subfolder with class directories exists within 'processed_data/quickdraw_raster/'.")
        return
    
    print(f"Training data will be loaded from: {TRAIN_DIR}")
    print(f"Validation data will be loaded from: {VAL_DIR}")
    if os.path.isdir(os.path.join(BASE_DATA_DIR, 'test')):
        print(f"Test data directory found at: {os.path.join(BASE_DATA_DIR, 'test')}")
        print("This script will only use train and validation sets. Testing can be done as a separate step.")


    # --- 2. Define data transformations ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # Convert to 3 channels for ResNet
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 3. Load datasets using ImageFolder ---
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transforms['train'])
        val_dataset = datasets.ImageFolder(VAL_DIR, data_transforms['val'])
    except Exception as e:
        print(f"An error occurred while loading datasets using ImageFolder: {e}")
        print("Please ensure your 'train' and 'val' directories are structured correctly,")
        print("with subfolders for each class containing the images.")
        return

    if not train_dataset.classes:
        print(f"Error: No classes found in {TRAIN_DIR}. Ensure subdirectories for each class exist.")
        return
    if not val_dataset.classes:
        print(f"Error: No classes found in {VAL_DIR}. Ensure subdirectories for each class exist.")
        return


    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Found {num_classes} classes in training set: {', '.join(class_names)}")

    # Verify that validation set has the same classes (optional but good practice)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("Warning: Training and validation sets have different class-to-index mappings. This might lead to issues.")
        print(f"Train classes: {train_dataset.class_to_idx}")
        print(f"Val classes: {val_dataset.class_to_idx}")
        # Consider re-mapping or ensuring consistency if this warning appears.
        # For simplicity, we'll proceed, assuming the order is what ImageFolder determines.

    # --- 4. Create DataLoaders ---
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print(f"Training set size: {dataset_sizes['train']}, Validation set size: {dataset_sizes['val']}")


    # --- 5. Load pre-trained ResNet50 model and modify the classifier ---
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features

    # Replace the last fully connected layer
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    # --- 6. Define loss function and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Optional: Learning rate scheduler
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 7. Train the model ---
    print("\nStarting training...")
    trained_model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS)

    # --- 8. Save the trained model ---
    print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}")
    try:
        torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25, scheduler=None):
    """
    Handles the training and validation loop for the model.
    (This function remains the same as in the previous version)
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'New best validation accuracy: {best_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        # Optional: for full determinism, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


    main()
