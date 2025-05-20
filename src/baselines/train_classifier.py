#!/usr/bin/env python3
"""
train_classifier_optimized.py (v5 – complete with debug mode)
---------------------------------------------
Fully optimized ResNet-50 sketch classifier training script with:
• cuDNN autotune + channels-last tensors
• AMP (FP16) or BF16 with GradScaler shim (PyTorch 1.12 → 2.2)
• Optional `torch.compile` for fused kernels on PyTorch ≥ 2.0
• Beefed-up DataLoader (persistent workers, prefetch)
• Single-pass profiler (first 4 batches)
• Robust checkpoint resume / best-model copy
• Debug mode: restrict to first N classes for quick smoke tests

Drop-in replacement for the original `train_classifier.py` – same CLI flags,
plus `--debug_num_classes`.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 🔧  Global switches
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import argparse
import shutil
import torch
from torch import nn, optim, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")  # PyTorch ≥2.1
except AttributeError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# 📂  Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# 🧩  Local imports
# ──────────────────────────────────────────────────────────────────────────────
try:
    from src.models.raster_encoder import ResNet50SketchEncoderBase
    from data.dataset import SketchDataset
except ImportError as e:
    sys.exit(f"Import error: {e}. Check repo layout.")

# ──────────────────────────────────────────────────────────────────────────────
# ⚙️  Defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_IMG_SIZE = 224
DEFAULT_ROOT     = os.path.join(PROJECT_ROOT, "processed_data")
DEFAULT_NAME     = "quickdraw"
DEFAULT_CFG      = "quickdraw_config.json"
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_WORKERS  = min(16, os.cpu_count() or 8)
DEFAULT_PREFETCH = 4

# ──────────────────────────────────────────────────────────────────────────────
# 🏗️  Model
# ──────────────────────────────────────────────────────────────────────────────
class SketchClassifier(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 1,
                 pretrained: bool = False, freeze: bool = False):
        super().__init__()
        self.backbone = ResNet50SketchEncoderBase(
            input_channels=in_ch,
            use_pretrained=pretrained,
            freeze_pretrained=freeze,
        )
        self.head = nn.Linear(self.backbone.output_feature_dim, num_classes)
        print(f"SketchClassifier → {num_classes} classes, feat {self.backbone.output_feature_dim}")

    def forward(self, x):
        return self.head(self.backbone(x))

# ──────────────────────────────────────────────────────────────────────────────
# 💾  Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_ckpt(epoch, model, opt, sched, hist, best, path, best_tag=False):
    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "best":  best,
        "hist":  hist,
        "sched": sched.state_dict() if sched else None,
    }, path)
    if best_tag:
        shutil.copyfile(path, path.replace("_latest_checkpoint", "_best_model"))

def load_ckpt(model, opt, sched, path, device):
    if not os.path.exists(path):
        print("No checkpoint — fresh start.")
        return 0, {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}, 0.0
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    if sched and ck.get("sched"):
        sched.load_state_dict(ck["sched"])
    return ck["epoch"], ck["hist"], ck["best"]

# ──────────────────────────────────────────────────────────────────────────────
# 🚂  Training / validation
# ──────────────────────────────────────────────────────────────────────────────
def train_validate(model, train_ld, val_ld, crit, opt, sched, epochs, device,
                   exp="Run", amp_flag=False, ckpt_dir="ckpts", save_every=1,
                   start_ep=0, hist=None, best_acc=0.0, profile=False):
    hist = hist or {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, f"{exp}_latest_checkpoint.pth")

    # ── GradScaler shim ─────────────────────────────────────────────
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        Scaler = torch.amp.GradScaler
        if device.type == "cuda" and amp_flag:
            scaler = Scaler(enabled=True)
        else:
            scaler = Scaler(enabled=False)
    else:
        from torch.cuda.amp import GradScaler as Scaler
        scaler = Scaler(enabled=amp_flag and device.type == "cuda")

    # helpers
    def prep(batch):
        x = batch["raster_image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = batch["label"].to(device, non_blocking=True)
        return x, y

    def train_step(batch):
        x, y = prep(batch)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_flag and device.type == "cuda"):
            out  = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss.item(), (out.argmax(1) == y).sum().item(), y.size(0)

    # profiler
    prof = None
    if profile and start_ep == 0:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=acts,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(ckpt_dir, f"{exp}_profile")),
            record_shapes=True, profile_memory=True, with_stack=True,
        )

    # epoch loop
    for ep in range(start_ep, epochs):
        ei = ep + 1
        model.train(); tloss = tcorrect = ttotal = 0
        pbar = tqdm(
            train_ld,
            desc=f"Ep {ei}/{epochs} [train]",
            leave=True,
            disable=False,
            file=sys.stderr,
        )

        # optional profiler block
        if prof and ep == start_ep:
            with prof:
                for i, batch in enumerate(pbar):
                    l, c, n = train_step(batch)
                    tloss += l * n; tcorrect += c; ttotal += n
                    prof.step()
                    if i >= 4: break
            for batch in pbar:
                l, c, n = train_step(batch)
                tloss += l * n; tcorrect += c; ttotal += n
        else:
            for batch in pbar:
                l, c, n = train_step(batch)
                tloss += l * n; tcorrect += c; ttotal += n

        tr_loss = tloss / ttotal; tr_acc = tcorrect / ttotal
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)

        # validation
        model.eval(); vloss = vcorr = vtot = 0
        with torch.no_grad():
            for batch in tqdm(val_ld, desc=f"Ep {ei}/{epochs} [val]", leave=False):
                x, y = prep(batch)
                with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_flag and device.type == "cuda"):
                    out = model(x); loss = crit(out, y)
                vloss += loss.item() * y.size(0); vcorr += (out.argmax(1) == y).sum().item(); vtot += y.size(0)
        va_loss = vloss / vtot; va_acc = vcorr / vtot
        hist["val_loss"].append(va_loss); hist["val_acc"].append(va_acc)

        # scheduler / log / ckpt
        if sched:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(va_loss)
            else:
                sched.step()
        print(f"Ep {ei}/{epochs} TL {tr_loss:.4f} TA {tr_acc:.4f} | VL {va_loss:.4f} VA {va_acc:.4f} | LR {opt.param_groups[0]['lr']:.2e}")

        best_tag = va_acc > best_acc
        if best_tag: best_acc = va_acc
        if ei % save_every == 0 or best_tag or ei == epochs:
            save_ckpt(ep, model, opt, sched, hist, best_acc, latest, best_tag)

    print(f"Training complete — best val acc {best_acc:.4f}")
    return hist

# ──────────────────────────────────────────────────────────────────────────────
# 🚀  Main
# ──────────────────────────────────────────────────────────────────────────────
def main(a):
    """Entry point built to be CLI-friendly."""
    device = torch.device(f"cuda:{a.gpu_id}" if a.use_gpu and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ── Transforms & channels ───────────────────────────────────────────
    tf = transforms.ToTensor(); in_ch = 1
    if a.use_pretrained:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        in_ch = 3

    # ── Datasets & loaders ──────────────────────────────────────────────
    root = os.path.abspath(a.dataset_root)
    trds = SketchDataset(root, a.dataset_name, split=a.train_split, image_size=a.img_size,
                         max_seq_len=a.max_seq_len, config_filename=a.config_filename,
                         raster_transform=tf)
    vlds = SketchDataset(root, a.dataset_name, split=a.val_split, image_size=a.img_size,
                         max_seq_len=a.max_seq_len, config_filename=a.config_filename,
                         raster_transform=tf)

    # ── Debug: restrict to first N classes ───────────────────────────────
    if a.debug_num_classes and a.debug_num_classes > 0:
        max_c = a.debug_num_classes
        tr_idxs = [i for i in range(len(trds)) if trds[i]['label'].item() < max_c]
        val_idxs = [i for i in range(len(vlds)) if vlds[i]['label'].item() < max_c]
        print(f"🛠 Debug mode: restricting to first {max_c} classes → "
              f"{len(tr_idxs)} train / {len(val_idxs)} val samples")
        trds = Subset(trds, tr_idxs)
        vlds = Subset(vlds, val_idxs)

    trld = DataLoader(trds, batch_size=a.batch_size, shuffle=True,
                      num_workers=a.num_workers or DEFAULT_WORKERS,
                      prefetch_factor=DEFAULT_PREFETCH, persistent_workers=True,
                      pin_memory=True)
    vld  = DataLoader(vlds, batch_size=a.batch_size, shuffle=False,
                      num_workers=a.num_workers or DEFAULT_WORKERS,
                      prefetch_factor=DEFAULT_PREFETCH, persistent_workers=True,
                      pin_memory=True)

    # ── Number of classes ───────────────────────────────────────────────
    if a.num_classes:
        ncls = a.num_classes
    elif hasattr(trds, 'num_classes') and trds.num_classes:
        ncls = trds.num_classes
    else:
        cfgp = os.path.join(root, f"{a.dataset_name}_vector", a.config_filename)
        if os.path.exists(cfgp):
            ncls = len(json.load(open(cfgp))["category_map"])
        else:
            raise ValueError("Could not infer num_classes. Provide --num_classes or a config with category_map.")

    # ── Model ───────────────────────────────────────────────────────────
    model = SketchClassifier(ncls, in_ch, in_ch==3 and a.use_pretrained, a.freeze_backbone)
    model = model.to(device, memory_format=torch.channels_last)
    if torch.__version__.startswith("2") and a.torch_compile:
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled with torch.compile().")

    # ── Loss, optim, sched ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    lr = a.lr * (0.1 if a.use_pretrained else 1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=a.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=a.lr_patience, factor=0.1)

    # ── Resume checkpoint (optional) ─────────────────────────────────────
    ckpt_dir = os.path.abspath(a.checkpoint_dir)
    latest   = os.path.join(ckpt_dir, f"{a.experiment_name}_latest_checkpoint.pth")
    start_ep, hist, best = (0, None, 0.0)
    if a.resume:
        start_ep, hist, best = load_ckpt(model, optimizer, scheduler, latest, device)

    # ── Train! ──────────────────────────────────────────────────────────
    train_validate(model, trld, vld, criterion, optimizer, scheduler,
                   epochs=a.epochs, device=device, exp=a.experiment_name,
                   amp_flag=a.use_amp, ckpt_dir=ckpt_dir, save_every=a.save_every,
                   start_ep=start_ep, hist=hist, best_acc=best, profile=a.profile)

# ──────────────────────────────────────────────────────────────────────────────
# 🛠️  CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Train ResNet-50 sketch classifier (optimized)")
    p.add_argument('--dataset_root',   default=DEFAULT_ROOT)
    p.add_argument('--dataset_name',   default=DEFAULT_NAME)
    p.add_argument('--config_filename',default=DEFAULT_CFG)
    p.add_argument('--train_split',    default='train')
    p.add_argument('--val_split',      default='val')
    p.add_argument('--img_size',       type=int,   default=DEFAULT_IMG_SIZE)
    p.add_argument('--max_seq_len',    type=int,   default=70)
    p.add_argument('--num_classes',    type=int)
    p.add_argument('--use_pretrained', action='store_true')
    p.add_argument('--freeze_backbone',action='store_true')
    p.add_argument('--epochs',         type=int,   default=20)
    p.add_argument('--batch_size',     type=int,   default=512)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--lr_patience',    type=int,   default=3)
    p.add_argument('--num_workers',    type=int)
    p.add_argument('--use_gpu',        action='store_true')
    p.add_argument('--gpu_id',         type=int,   default=0)
    p.add_argument('--use_amp',        action='store_true')
    p.add_argument('--torch_compile',  action='store_true')
    p.add_argument('--checkpoint_dir', default=DEFAULT_CKPT_DIR)
    p.add_argument('--resume',         action='store_true')
    p.add_argument('--save_every',     type=int,   default=1)
    p.add_argument('--profile',        action='store_true')
    p.add_argument(
        '--debug_num_classes',
        type=int,
        default=0,
        help="If >0, only load samples with label < this (for quick debug)"
    )
    main(p.parse_args())
