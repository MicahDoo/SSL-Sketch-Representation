#!/usr/bin/env python3
"""
train_classifier_optimized.py (v3)
----------------------------------
* Fixes GradScaler import/signature clash (PyTorch < 2 vs 2).
* Chooses `torch.amp.GradScaler` when available, otherwise falls back to
  `torch.cuda.amp.GradScaler` (no `device_type` kwarg) — no more TypeError.
"""

# -----------------------------------------------------------------------------
# 🔧  Global performance switches
# -----------------------------------------------------------------------------
import os, sys, json, time, argparse, shutil, torch
from torch import nn, optim, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# 📂  Project paths
# -----------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# 🧩  Local imports
# -----------------------------------------------------------------------------
try:
    from src.models.raster_encoder import ResNet50SketchEncoderBase
    from data.dataset import SketchDataset
except ImportError as e:
    sys.exit(f"Import error: {e}. Check repo layout.")

# -----------------------------------------------------------------------------
# ⚙️  Defaults
# -----------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 224
DEFAULT_ROOT     = os.path.join(PROJECT_ROOT, "processed_data")
DEFAULT_NAME     = "quickdraw"
DEFAULT_CFG      = "quickdraw_config.json"
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_WORKERS  = min(16, os.cpu_count() or 8)
DEFAULT_PREFETCH = 4

# -----------------------------------------------------------------------------
# 🏗️  Model
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 💾  Checkpoint helpers
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 🚂  Training / validation
# -----------------------------------------------------------------------------

def train_validate(model, train_ld, val_ld, crit, opt, sched, epochs, device,
                   exp="Run", amp_flag=False, ckpt_dir="ckpts", save_every=1,
                   start_ep=0, hist=None, best_acc=0.0, profile=False):
    hist = hist or {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, f"{exp}_latest_checkpoint.pth")

    # --- GradScaler selection (torch 1.x vs 2.x) ----------------------------
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        ScalerCls = torch.amp.GradScaler  # PyTorch ≥2.0
        scaler    = ScalerCls(enabled=amp_flag and device.type == "cuda",
                              device_type=device.type)
    else:
        from torch.cuda.amp import GradScaler as ScalerCls  # PyTorch 1.x fallback
        scaler    = ScalerCls(enabled=amp_flag and device.type == "cuda")

    # --- helpers ------------------------------------------------------------
    def prep(batch):
        x = batch["raster_image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = batch["label"].to(device, non_blocking=True)
        return x, y

    def train_step(batch):
        x, y = prep(batch)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_flag and device.type=="cuda"):
            out  = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss.item(), (out.argmax(1) == y).sum().item(), y.size(0)

    # --- profiler (optional) -------------------------------------------------
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

    # --- epoch loop ---------------------------------------------------------
    for ep in range(start_ep, epochs):
        ei = ep + 1
        model.train(); tloss = tcorrect = ttotal = 0
        pbar = tqdm(train_ld, desc=f"Ep {ei}/{epochs} [train]", leave=False)

        if prof and ep == start_ep:
            with prof:
                for i, batch in enumerate(pbar):
                    l, c, n = train_step(batch)
                    tloss += l * n; tcorrect += c; ttotal += n
                    prof.step()
                    if i >= 4: break
            for batch in pbar:  # continue remainder of epoch
                l, c, n = train_step(batch)
                tloss += l * n; tcorrect += c; ttotal += n
        else:
            for batch in pbar:
                l, c, n = train_step(batch)
                tloss += l * n; tcorrect += c; ttotal += n

        tr_loss = tloss/ttotal; tr_acc = tcorrect/ttotal
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)

        # --- validation ------------------------------------------------------
        model.eval(); vloss = vcorrect = vtotal = 0
        with torch.no_grad():
            for batch in tqdm(val_ld, desc=f"Ep {ei}/{epochs} [val]", leave=False):
                x,y = prep(batch)
                with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_flag and device.type=="cuda"):
                    out = model(x); loss = crit(out, y)
                vloss += loss.item()*y.size(0); vcorrect += (out.argmax(1)==y).sum().item(); vtotal += y.size(0)
        va_loss = vloss/vtotal; va_acc = vcorrect/vtotal
        hist["val_loss"].append(va_loss); hist["val_acc"].append(va_acc)

        # --- scheduler / logging / ckpt -------------------------------------
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

# -----------------------------------------------------------------------------
# 🚀  Main
# -----------------------------------------------------------------------------

def main(a):
    device = torch.device(f"cuda:{a.gpu_id}" if a.use_gpu and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # transforms
    tf = transforms.ToTensor(); in_ch = 1
    if a.use_pretrained:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        in_ch = 3

    # datasets
    root = os.path.abspath(a.dataset_root)
    trds  = SketchDataset(root, a.dataset_name, split=a.train_split, image_size=a.img_size,
                          max_seq_len=a.max_seq_len, config_filename=a.config_filename, raster_transform=tf)
    vlds  = SketchDataset(root, a.dataset_name, split=a.val_split, image_size=a.img_size,
                          max_seq_len=a.max_seq_len, config_filename=a.config_filename, raster_transform=tf)

    trld = DataLoader(trds, batch_size=a.batch_size, shuffle=True,
                      num_workers=a.num_workers or DEFAULT_WORKERS,
                      prefetch_factor=DEFAULT_PREFETCH, persistent_workers=True, pin_memory=True)
    vld  = DataLoader(vlds, batch_size=a.batch_size, shuffle=False,
                      num_workers=a.num_workers or DEFAULT_WORKERS,
                      prefetch_factor=DEFAULT_PREFETCH, persistent_workers=True, pin_memory=True)

    # num classes
    if a.num_classes:
        ncls = a.num_classes
    elif hasattr(trds, "num_classes") and trds.num_classes:
        ncls = trds.num_classes
    else:
        cfgp = os.path.join(root, f"{a.dataset_name}_vector", a.config_filename)
        if os.path.exists(cfgp):
            ncls = len(json.load(open(cfgp))["category_map"])
        else:
            raise ValueError("num_classes undetermined")

    # model
    model = SketchClassifier(ncls, in_ch, in_ch==3 and a.use_pretrained, a.freeze_backbone)
    model = model.to(device, memory_format=torch.channels_last)
    if torch.__version__.startswith("2") and a.torch_compile:
