#!/usr/bin/env python3
"""
train_classifier_optimized.py
---------------------------------
A fully‑refactored version of the original `train_classifier.py`, with

* cuDNN autotune (`torch.backends.cudnn.benchmark`)
* channels‑last memory format on model **and** inputs
* Automatic Mixed Precision (FP16) via `torch.autocast`
* optional `torch.compile` for kernel fusion on PyTorch 2.x
* beefed‑up DataLoader (16 workers, prefetch, pinned memory, persistent workers)
* single‑pass profiling (no double compute)
* non‑blocking host→device transfers

The CLI is unchanged, so all existing shell scripts keep working.
"""

# -----------------------------------------------------------------------------
# 🔧 Global performance switches — must be set **before** the first model import
# -----------------------------------------------------------------------------
import os, sys, time, json, argparse, shutil
import torch

torch.backends.cudnn.benchmark = True                       # fastest conv kernels
torch.set_float32_matmul_precision('high')                  # better GEMMs on ≥2.1

# -----------------------------------------------------------------------------
# 📂 Path setup
# -----------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"DEBUG: Added {PROJECT_ROOT} to sys.path")

# -----------------------------------------------------------------------------
# 🧩 Imports that rely on `PROJECT_ROOT` being on sys.path
# -----------------------------------------------------------------------------
from torch import nn, optim, autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    from src.models.raster_encoder import ResNet50SketchEncoderBase
    from data.dataset import SketchDataset
except ImportError as e:
    sys.exit(f"Import error: {e}. Check your repo layout.")

# -----------------------------------------------------------------------------
# ⚙️ Configuration defaults
# -----------------------------------------------------------------------------
DEFAULT_SKETCH_IMG_SIZE = 224
DEFAULT_DATASET_ROOT    = os.path.join(PROJECT_ROOT, "processed_data")
DEFAULT_DATASET_NAME    = "quickdraw"
DEFAULT_CONFIG_FILENAME = "quickdraw_config.json"
DEFAULT_CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_NUM_WORKERS     = min(16, os.cpu_count() or 8)
DEFAULT_PREFETCH        = 4

# -----------------------------------------------------------------------------
# 🏗️ Model definition
# -----------------------------------------------------------------------------
class SketchClassifier(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 1,
                 use_pretrained_backbone: bool = False,
                 freeze_backbone: bool = False):
        super().__init__()
        self.backbone = ResNet50SketchEncoderBase(
            input_channels=input_channels,
            use_pretrained=use_pretrained_backbone,
            freeze_pretrained=freeze_backbone,
        )
        self.classifier_head = nn.Linear(self.backbone.output_feature_dim, num_classes)
        print(f"SketchClassifier → classes: {num_classes}, backbone dim: {self.backbone.output_feature_dim}")

    def forward(self, x):
        return self.classifier_head(self.backbone(x))

# -----------------------------------------------------------------------------
# 💾 Checkpoint helpers
# -----------------------------------------------------------------------------

def save_checkpoint(epoch, model, optimizer, scheduler, history, best_val_acc, path, is_best=False):
    ckpt = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
    }
    if scheduler:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)
    print(f"Checkpoint saved → {path}")
    if is_best:
        best_path = path.replace("_latest_checkpoint", "_best_model")
        shutil.copyfile(path, best_path)
        print(f"Best model copied → {best_path}")

def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        print("No checkpoint found — starting fresh.")
        return 0, {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from epoch {ckpt['epoch']} (best val_acc={ckpt['best_val_acc']:.4f})")
    return ckpt["epoch"], ckpt["history"], ckpt["best_val_acc"]

# -----------------------------------------------------------------------------
# 🚂 Training / validation loop
# -----------------------------------------------------------------------------

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler,
                       epochs, device, experiment_name="Classifier", use_amp=False,
                       ckpt_dir="checkpoints", save_every=1, start_epoch=0,
                       history=None, best_val_acc=0.0, profile_steps=False):
    history = history or {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt = os.path.join(ckpt_dir, f"{experiment_name}_latest_checkpoint.pth")

    amp_enabled = use_amp and device.type == "cuda"
    scaler      = GradScaler(enabled=amp_enabled)

    # Optional: profiler for first N steps
    prof = None
    if profile_steps and start_epoch == 0:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        trace_dir = os.path.join(ckpt_dir, f"{experiment_name}_profile")
        os.makedirs(trace_dir, exist_ok=True)
        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True, profile_memory=True, with_stack=True,
        )

    for epoch in range(start_epoch, epochs):
        ep = epoch + 1
        model.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]", leave=False)
        tr_loss = tr_correct = tr_total = 0

        # --------------------------------- TRAIN --------------------------------
        if prof and epoch == start_epoch:
            print("Profiler running for initial batches…")
            with prof:
                for b, batch in enumerate(train_loop):
                    _train_step(batch)
                    prof.step()
                    if b >= 4:  # 1 warm‑up + 3 active
                        break
            continue  # skip duplicate compute below

        for batch in train_loop:
            loss, correct, total = _train_step(batch)
            tr_loss += loss * total
            tr_correct += correct
            tr_total += total

        epoch_train_loss = tr_loss / tr_total
        epoch_train_acc  = tr_correct / tr_total
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        # --------------------------------- VAL ----------------------------------
        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [val]", leave=False):
                inputs, targets = _prep_batch(batch)
                with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(inputs)
                    loss    = criterion(outputs, targets)
                vl_loss   += loss.item() * targets.size(0)
                vl_correct += (outputs.argmax(1) == targets).sum().item()
                vl_total   += targets.size(0)
        epoch_val_loss = vl_loss / vl_total
        epoch_val_acc  = vl_correct / vl_total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # LR scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # Console report
        print(f"Epoch {ep}/{epochs} — "
              f"TrL {epoch_train_loss:.4f} • TrA {epoch_train_acc:.4f} | "
              f"VaL {epoch_val_loss:.4f} • VaA {epoch_val_acc:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.3e}")

        # Checkpoint
        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
        if ep % save_every == 0 or is_best or ep == epochs:
            save_checkpoint(epoch, model, optimizer, scheduler, history, best_val_acc,
                            latest_ckpt, is_best)

    print(f"Training done — best val acc: {best_val_acc:.4f}")
    return history

    # ---------------------------- inner helpers -----------------------------

    def _prep_batch(batch):
        inputs  = batch["raster_image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        inputs  = inputs.to(memory_format=torch.channels_last)
        return inputs, targets

    def _train_step(batch):
        inputs, targets = _prep_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        correct = (outputs.argmax(1) == targets).sum().item()
        return loss.item(), correct, targets.size(0)

# -----------------------------------------------------------------------------
# 🚀 Main
# -----------------------------------------------------------------------------

def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms (repeat grayscale → 3 channels if using ImageNet weights)
    tf = transforms.ToTensor()
    ch_in = 1
    if args.use_pretrained:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        ch_in = 3

    # Datasets & loaders
    ds_root = os.path.abspath(args.dataset_root)
    train_ds = SketchDataset(ds_root, args.dataset_name, split=args.train_split,
                             image_size=args.img_size, max_seq_len=args.max_seq_len,
                             config_filename=args.config_filename, raster_transform=tf)
    val_ds   = SketchDataset(ds_root, args.dataset_name, split=args.val_split,
                             image_size=args.img_size, max_seq_len=args.max_seq_len,
                             config_filename=args.config_filename, raster_transform=tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers or DEFAULT_NUM_WORKERS,
                              prefetch_factor=DEFAULT_PREFETCH, persistent_workers=True,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers or DEFAULT_NUM_WORKERS,
                              prefetch_factor=DEFAULT_PREFETCH, persistent_workers=True,
                              pin_memory=True)

        # ------- Number of classes -------
    if args.num_classes:
        num_classes = args.num_classes
    elif hasattr(train_ds, 'num_classes') and train_ds.num_classes:
        num_classes = train_ds.num_classes
        print(f"Number of classes determined from train_ds: {num_classes}")
    else:
        config_path = os.path.join(ds_root, f"{args.dataset_name}_vector", args.config_filename)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            num_classes = len(cfg.get('category_map', {}))
            print(f"Loaded {num_classes} classes from {config_path}")
        else:
            raise ValueError("Could not infer num_classes. Pass --num_classes or supply a config with category_map.")

    # Model — channels_last + (optional) compile
    model = SketchClassifier(num_classes, input_channels=ch_in,
                             use_pretrained_backbone=args.use_pretrained,
                             freeze_backbone=args.freeze_backbone)
    model = model.to(device, memory_format=torch.channels_last)
    if torch.__version__.startswith("2") and args.torch_compile:
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled with torch.compile().")

    # Loss, opt, sched
    criterion = nn.CrossEntropyLoss()
    lr        = args.lr if not args.use_pretrained else args.lr * 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",
                                                    patience=args.lr_patience, factor=0.1)

    # Resume
    ckpt_dir  = os.path.abspath(args.checkpoint_dir)
    latest_ckpt = os.path.join(ckpt_dir, f"ResNetClassify_latest_checkpoint.pth")
    start_epoch, history, best_acc = (0, None, 0.0)
    if args.resume:
        start_epoch, history, best_acc = load_checkpoint(model, optimizer, scheduler, latest_ckpt, device)

    # Train
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler,
                       epochs=args.epochs, device=device, experiment_name="ResNetClassify",
                       use_amp=args.use_amp, ckpt_dir=ckpt_dir, save_every=args.save_every,
                       start_epoch=start_epoch, history=history, best_val_acc=best_acc,
                       profile_steps=args.profile)

# -----------------------------------------------------------------------------
# 🛠️ CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser("Train a ResNet sketch classifier (optimized)")
    p.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    p.add_argument("--config_filename", default=DEFAULT_CONFIG_FILENAME)
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split",   default="val")
    p.add_argument("--img_size", type=int, default=DEFAULT_SKETCH_IMG_SIZE)
    p.add_argument("--max_seq_len", type=int, default=70)
    p.add_argument("--num_classes", type=int)
    p.add_argument("--use_pretrained", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_patience", type=int, default=3)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--torch_compile", action="store_true")
    p.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--profile", action="store_true")

    main(p.parse_args())
