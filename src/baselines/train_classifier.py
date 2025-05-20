#!/usr/bin/env python3
"""
train_classifier_optimized.py (v2)
----------------------------------
* Fixes `_train_step` scope error (helpers defined before use)
* Future‑proof AMP: `torch.amp.GradScaler(device_type='cuda', ...)`
* Profiler no longer skips the rest of the epoch; it profiles first 4 batches and then keeps training.
"""

# -----------------------------------------------------------------------------
# 🔧 Global performance switches
# -----------------------------------------------------------------------------
import os, sys, json, time, argparse, shutil, torch
from torch import nn, optim, autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# -----------------------------------------------------------------------------
# 📂 Project paths
# -----------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# 🧩 Local imports
# -----------------------------------------------------------------------------
try:
    from src.models.raster_encoder import ResNet50SketchEncoderBase
    from data.dataset import SketchDataset
except ImportError as e:
    sys.exit(f"Import error: {e}. Check repo layout.")

# -----------------------------------------------------------------------------
# ⚙️ Defaults
# -----------------------------------------------------------------------------
DEFAULT_IMG_SIZE   = 224
DEFAULT_ROOT       = os.path.join(PROJECT_ROOT, "processed_data")
DEFAULT_NAME       = "quickdraw"
DEFAULT_CFG        = "quickdraw_config.json"
DEFAULT_CKPT_DIR   = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_WORKERS    = min(16, os.cpu_count() or 8)
DEFAULT_PREFETCH   = 4

# -----------------------------------------------------------------------------
# 🏗️ Model
# -----------------------------------------------------------------------------
class SketchClassifier(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 1,
                 pretrained: bool = False, freeze: bool = False):
        super().__init__()
        self.backbone = ResNet50SketchEncoderBase(input_channels=in_ch,
                                                  use_pretrained=pretrained,
                                                  freeze_pretrained=freeze)
        self.head = nn.Linear(self.backbone.output_feature_dim, num_classes)
        print(f"SketchClassifier → {num_classes} classes, feat {self.backbone.output_feature_dim}")

    def forward(self, x):
        return self.head(self.backbone(x))

# -----------------------------------------------------------------------------
# 💾 Checkpoint helpers
# -----------------------------------------------------------------------------

def save_ckpt(epoch, model, opt, sched, hist, best, path, best_tag=False):
    ckpt = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "best":  best,
        "hist":  hist,
    }
    if sched:
        ckpt["sched"] = sched.state_dict()
    torch.save(ckpt, path)
    if best_tag:
        shutil.copyfile(path, path.replace("_latest_checkpoint", "_best_model"))


def load_ckpt(model, opt, sched, path, device):
    if not os.path.exists(path):
        print("No checkpoint — fresh start.")
        return 0, {k: [] for k in ("train_loss","train_acc","val_loss","val_acc")}, 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    if sched and "sched" in ckpt:
        sched.load_state_dict(ckpt["sched"])
    return ckpt["epoch"], ckpt["hist"], ckpt["best"]

# -----------------------------------------------------------------------------
# 🚂 Training
# -----------------------------------------------------------------------------

def train_validate(model, train_ld, val_ld, crit, opt, sched, epochs, device,
                   exp="Run", amp_flag=False, ckpt_dir="ckpts", save_every=1,
                   start_ep=0, hist=None, best_acc=0.0, profile=False):
    hist = hist or {k: [] for k in ("train_loss","train_acc","val_loss","val_acc")}
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, f"{exp}_latest_checkpoint.pth")

    amp = amp_flag and device.type == "cuda"
    scaler = GradScaler(enabled=amp, device_type="cuda" if device.type=="cuda" else "cpu")

    # ---- helpers (defined before use) ---------------------------------------
    def prep(batch):
        x  = batch["raster_image"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y  = batch["label"].to(device, non_blocking=True)
        return x, y

    def train_step(batch):
        x, y = prep(batch)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        correct = (out.argmax(1)==y).sum().item()
        return loss.item(), correct, y.size(0)

    # ---- optional profiler --------------------------------------------------
    prof = None
    if profile and start_ep == 0:
        acts=[torch.profiler.ProfilerActivity.CPU]
        if device.type=="cuda": acts.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=acts,
            schedule=torch.profiler.schedule(wait=0,warmup=1,active=3,repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(ckpt_dir, f"{exp}_profile")),
            record_shapes=True, profile_memory=True, with_stack=True)

    # ---- epoch loop ---------------------------------------------------------
    for ep in range(start_ep, epochs):
        ei = ep+1
        model.train()
        tbar = tqdm(train_ld, desc=f"Ep {ei}/{epochs} [train]", leave=False)
        tloss=tcorrect=ttotal=0

        if prof and ep==start_ep:
            with prof:
                for b, batch in enumerate(tbar):
                    loss, cor, tot = train_step(batch)
                    tloss+=loss*tot; tcorrect+=cor; ttotal+=tot
                    prof.step()
                    if b>=4: break  # profile first 4 batches
            # continue training same epoch for remaining batches
            for batch in tbar:
                loss, cor, tot = train_step(batch)
                tloss+=loss*tot; tcorrect+=cor; ttotal+=tot
        else:
            for batch in tbar:
                loss, cor, tot = train_step(batch)
                tloss+=loss*tot; tcorrect+=cor; ttotal+=tot

        tr_loss = tloss/ttotal; tr_acc = tcorrect/ttotal
        hist["train_loss"].append(tr_loss); hist["train_acc"].append(tr_acc)

        # ---- validation ------------------------------------------------------
        model.eval(); vloss=vcorr=vtot=0
        with torch.no_grad():
            for batch in tqdm(val_ld, desc=f"Ep {ei}/{epochs} [val]", leave=False):
                x,y = prep(batch)
                with autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    out = model(x); loss = crit(out,y)
                vloss+=loss.item()*y.size(0); vcorr+=(out.argmax(1)==y).sum().item(); vtot+=y.size(0)
        va_loss=vloss/vtot; va_acc=vcorr/vtot
        hist["val_loss"].append(va_loss); hist["val_acc"].append(va_acc)

        # --- sched, print, ckpt ----------------------------------------------
        if sched:
            (sched.step(va_loss) if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
             else sched.step())
        print(f"Ep{ei}/{epochs}  TL {tr_loss:.4f} TA {tr_acc:.4f} | VL {va_loss:.4f} VA {va_acc:.4f} | LR {opt.param_groups[0]['lr']:.2e}")

        best_tag=False
        if va_acc>best_acc: best_acc=va_acc; best_tag=True
        if ei%save_every==0 or best_tag or ei==epochs:
            save_ckpt(ep, model, opt, sched, hist, best_acc, latest, best_tag)

    print(f"Training complete — best val acc {best_acc:.4f}")
    return hist

# -----------------------------------------------------------------------------
# 🚀 Main
# -----------------------------------------------------------------------------

def main(a):
    device = torch.device(f"cuda:{a.gpu_id}" if a.use_gpu and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # transform & channels
    tf=transforms.ToTensor(); in_ch=1
    if a.use_pretrained:
        tf=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]); in_ch=3

    # datasets
    root=os.path.abspath(a.dataset_root)
    trds=SketchDataset(root,a.dataset_name,split=a.train_split,image_size=a.img_size,
                       max_seq_len=a.max_seq_len,config_filename=a.config_filename,raster_transform=tf)
    vlds=SketchDataset(root,a.dataset_name,split=a.val_split,image_size=a.img_size,
                       max_seq_len=a.max_seq_len,config_filename=a.config_filename,raster_transform=tf)

    trld=DataLoader(trds,batch_size=a.batch_size,shuffle=True,num_workers=a.num_workers or DEFAULT_WORKERS,
                    prefetch_factor=DEFAULT_PREFETCH,persistent_workers=True,pin_memory=True)
    vld=DataLoader(vlds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers or DEFAULT_WORKERS,
                   prefetch_factor=DEFAULT_PREFETCH,persistent_workers=True,pin_memory=True)

    # classes
    if a.num_classes: ncls=a.num_classes
    elif hasattr(trds,'num_classes') and trds.num_classes: ncls=trds.num_classes
    else:
        cfgp=os.path.join(root,f"{a.dataset_name}_vector",a.config_filename)
        if os.path.exists(cfgp): ncls=len(json.load(open(cfgp))['category_map'])
        else: raise ValueError("num_classes undetermined")

    # model
    model=SketchClassifier(ncls,in_ch,in_ch==3 and a.use_pretrained,a.freeze_backbone)
    model=model.to(device,memory_format=torch.channels_last)
    if torch.__version__.startswith("2") and a.torch_compile:
        model=torch.compile(model,mode="max-autotune")

    crit=nn.CrossEntropyLoss()
    lr=a.lr*(0.1 if a.use_pretrained else 1)
    opt=optim.Adam(model.parameters(),lr=lr,weight_decay=a.weight_decay)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=a.lr_patience,factor=0.1)

    ckdir=os.path.abspath(a.checkpoint_dir); latest=os.path.join(ckdir,"ResNetClassify_latest_checkpoint.pth")
    st_ep,hist,best=(0,None,0.0)
    if a.resume: st_ep,hist,best=load_ckpt(model,opt,sched,latest,device)

    train_validate(model,trld,vld,crit,opt,sched,a.epochs,device,exp="ResNetClassify",
                   amp_flag=a.use_amp,ckpt_dir=ckdir,save_every=a.save_every,start_ep=st_ep,
                   hist=hist,best_acc=best,profile=a.profile)

# -----------------------------------------------------------------------------
# 🛠️ CLI
# -----------------------------------------------------------------------------

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset_root",default=DEFAULT_ROOT)
    p.add_argument("--dataset_name",default=DEFAULT_NAME)
    p.add_argument("--config_filename",default=DEFAULT_CFG)
    p.add_argument("--train_split",default="train")
    p.add_argument("--val_split",default="val")
    p.add_argument("--img_size",type=int,default=DEFAULT_IMG_SIZE)
    p.add_argument("--max_seq_len",type=int,default=70)
    p.add_argument("--num_classes",type=int)
    p.add_argument("--use_pretrained",action="store_true")
    p.add_argument("--freeze_backbone",action="store_true")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch_size",type=int,default=512)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight_decay",type=float,default=1e-4)
    p.add_argument("--lr_patience",type=int,default=3)
    p.add_argument("--num_workers",type=int)
    p.add_argument("--use_gpu",action="store_true")
    p.add_argument("--gpu_id",type=int,default=0)
    p.add_argument("--use_amp",action="store_true")
    p.add_argument("--torch_compile",action="store_true")
    p.add_argument("--checkpoint_dir",default=DEFAULT_CKPT_DIR)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--save_every",type=int,default=1)
    p.add_argument("--profile",action="store_true")

    main(p.parse_args())
