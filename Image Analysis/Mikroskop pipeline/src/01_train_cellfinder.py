import os
import argparse
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from bioio import BioImage
from tqdm import tqdm
import time
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler #für weighted sampler bei picture augmentation through rotation

from cellSAM.sam_inference import CellSAM
from cellSAM.AnchorDETR.models.anchor_detr import SetCriterion
from cellSAM.AnchorDETR.models.matcher import build_matcher

# -------------------------
# IO helpers
# -------------------------


def build_sampler_weights(dataset, aug_weight=0.3, aug_marker="_aug"):
    weights = []
    for uid, _g in dataset.groups:
        weights.append(aug_weight if aug_marker in str(uid) else 1.0)
    return torch.as_tensor(weights, dtype=torch.double)


def read_image_any(path):
    """
    Returns a 2D numpy array (Y, X).
    - .tif/.tiff: reads as-is (if it’s already 2D MIP)
    - .czi: loads and max-projects over Z to produce a 2D MIP for the chosen (T, C, Scene)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(path)
        arr = np.squeeze(arr)
        # If someone accidentally gives you (Z,Y,X), you can still MIP:
        if arr.ndim == 3:
            arr = arr.max(axis=0)
        if arr.ndim != 2:
            raise ValueError(f"TIFF not 2D after squeeze/MIP. shape={arr.shape} path={path}")
        return arr

    if ext == ".czi":
        
        img = BioImage(path)
        data = img.data

        d = np.asarray(data)

        if d.ndim == 5:      # T C Z Y X
            zyx = d[0, 0]
        elif d.ndim == 4:    # C Z Y X  or  T Z Y X
            zyx = d[0]
        elif d.ndim == 3:    # Z Y X
            zyx = d
        else:
            raise ValueError(f"Unsupported CZI array ndim={d.ndim}, shape={d.shape}, path={path}")

        mip = zyx.max(axis=0)  # max over Z -> (Y,X)
        return mip.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported extension: {ext} for {path}")

def mip_to_uint8_rgb(img2d: np.ndarray, p_low=1.0, p_high=99.0) -> torch.Tensor:
    
    x = img2d.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    u8 = (x * 255.0).astype(np.uint8)
    rgb = np.stack([u8, u8, u8], axis=0)  # (3,H,W)
    
    return torch.from_numpy(rgb)

def xyxy_to_cxcywh_norm(g, W, H):
    
    # targets (normalized cxcywh in [0,1])
    cx = ((g["xmin"] + g["xmax"]) / 2.0).to_numpy(np.float32) / W
    cy = ((g["ymin"] + g["ymax"]) / 2.0).to_numpy(np.float32) / H
    w  = (g["xmax"] - g["xmin"]).to_numpy(np.float32) / W
    h  = (g["ymax"] - g["ymin"]).to_numpy(np.float32) / H
    boxes = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
    boxes = np.clip(boxes, 0.0, 1.0)
    
    return boxes

def split_by_uid(csv_path: str, val_frac: float = 0.2, test_frac: float = 0.1, seed: int = 6):
    
    df = pd.read_csv(csv_path, sep="\t")
    df = df.dropna(subset=["uid","image_path","xmin","ymin","xmax","ymax"]).copy()
    
    uids = df["uid"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(uids)

    n = len(uids)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split fractions too large; train set would be empty.")

    train_u = set(uids[:n_train])
    val_u   = set(uids[n_train:n_train+n_val])
    test_u  = set(uids[n_train+n_val:])

    train_df = df[df["uid"].isin(train_u)].copy()
    val_df   = df[df["uid"].isin(val_u)].copy()
    test_df  = df[df["uid"].isin(test_u)].copy()
    return train_df, val_df, test_df


# -------------------------
# Dataset
# -------------------------
class CellFinderDataset(Dataset):
    def __init__(self, bboxes_csv, resize=1024):
        
        self.df = pd.read_csv(bboxes_csv, sep="\t", engine="python")  # handles csv/tsv
        self.groups = list(self.df.groupby("uid"))
        self.resize = resize

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        
        _, g = self.groups[idx]
        img_path = g["image_path"].iloc[0]
        img = read_image_any(img_path)
    
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got {img.shape} at {img_path}")

        H, W = img.shape
        x = mip_to_uint8_rgb(img)  # (3,H,W), uint8 in [0,255]

        # targets
        boxes = xyxy_to_cxcywh_norm(g, W=W, H=H)
        # If you only have 1 semantic class ("cell"), pick ONE label index and use it consistently.
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)  # class 0 for all cells

        target = {
            "labels": torch.from_numpy(labels),
            "boxes": torch.from_numpy(boxes),
        }
        
        return x, target
    
def collate_fn(batch):
    xs, targets = zip(*batch)
    xs = torch.stack(xs, dim=0)
    return xs, list(targets)

# -------------------------
# Loss plotting + logging
# -------------------------
def append_loss_row(loss_tsv: str, row: dict):
    header_needed = not os.path.exists(loss_tsv)
    df = pd.DataFrame([row])
    df.to_csv(loss_tsv, sep="\t", index=False, mode="a", header=header_needed)

def plot_losses(loss_tsv: str, out_png: str):
    if not os.path.exists(loss_tsv):
        return
    df = pd.read_csv(loss_tsv, sep="\t")
    if df.empty:
        return
    plt.figure(figsize=(7,4))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
def add_aux_weights(weight_dict, num_dec_layers=6):
    # mirror DETR: add loss_x_i for intermediate decoder layers
    aux = {}
    for i in range(num_dec_layers - 1):
        for k, v in weight_dict.items():
            aux[f"{k}_{i}"] = v
    weight_dict.update(aux)
    return weight_dict

    
# -------------------------
# Training and Eval loops
# -------------------------
def train_one_epoch(cellfinder, cellsam, train_loader, criterion, optimizer, device, use_amp=True):
    
    cellfinder.train()
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.startswith("cuda")))
    
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    for xs, targets in tqdm(train_loader, leave=False, desc="Train"):
        xs = cellsam.sam_bbox_preprocessing(list(xs), percentile=False)  # (B,3,1024,1024) on device
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        with torch.amp.autocast('cuda', enabled=(use_amp and device.startswith("cuda"))):
            out = cellfinder(xs)
            loss_dict = criterion(out, targets)
            loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
            
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(cellfinder.parameters(), 0.1)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        
    return running_loss / len(train_loader)

def evaluate(cellfinder, cellsam, val_loader, criterion, device, use_amp=True):
    
    cellfinder.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for xs, targets in tqdm(val_loader, leave=False, desc="Eval"):
            xs = cellsam.sam_bbox_preprocessing(list(xs), percentile=False)  # (B,3,1024,1024) on device
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            with torch.amp.autocast('cuda', enabled=(use_amp and device.startswith("cuda"))):
                out = cellfinder(xs)
                loss_dict = criterion(out, targets)
                loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)

            running_loss += loss.item()
            
    return running_loss / len(val_loader)

def set_backbone_trainable(decode_head, trainable: bool):
    """
    Toggle requires_grad for the detector backbone (SAMBackbone inside AnchorDETR).
    In the repo model, the backbone is under decode_head.backbone.
    """
    bb = getattr(decode_head, "backbone", None)
    if bb is None:
        print("[WARN] decode_head has no attribute 'backbone' - cannot freeze/unfreeze.")
        return
    for p in bb.parameters():
        p.requires_grad = trainable

# -------------------------
# Main
# -------------------------
def parse_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--bboxes_csv", required=False, default=None)   # war required=True
    ap.add_argument("--use_existing_splits", action="store_true",
                    help="Nutze vorhandene train/val/test_bboxes.tsv statt intern zu splitten.")
    ap.add_argument("--split_dir", default=None, help="Ordner mit den TSVs. Default: --outdir.")
    ap.add_argument("--init_cellsam_ckpt", default=None,
                    help="Optional: path to a CellSAM .pt checkpoint to initialize from (recommended).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0,
                    help="Freeze detector backbone for first N epochs, then unfreeze.")
    ap.add_argument("--num_workers", required=False, type=int, default=8)
    args = ap.parse_args()

    
    return args

def main():

    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    loss_tsv = os.path.join(args.outdir, "losses.tsv")
    loss_png = os.path.join(args.outdir, "loss_curve.png")
    best_ckpt = os.path.join(args.outdir, "cellfinder_ft6_ckpt.pt")

    # Load datasets
    if args.use_existing_splits:
        split_dir = args.split_dir or args.outdir
        train_tsv = os.path.join(split_dir, "train_bboxes.tsv")
        val_tsv   = os.path.join(split_dir, "val_bboxes.tsv")
        print(f"[INFO] Nutze vorhandene Splits aus: {split_dir}")
    else:
        train_df, val_df, test_df = split_by_uid(csv_path=args.bboxes_csv,
                                                 val_frac=args.val_frac,
                                                 test_frac=args.test_frac, seed=42)
        train_tsv = f"{args.outdir}/train_bboxes.tsv"
        val_tsv   = f"{args.outdir}/val_bboxes.tsv"
        train_df.to_csv(train_tsv, sep="\t", index=False)
        val_df.to_csv(val_tsv, sep="\t", index=False)
        test_df.to_csv(f"{args.outdir}/test_bboxes.tsv", sep="\t", index=False)
        print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_dset = CellFinderDataset(train_tsv, resize=1024)
    val_dset   = CellFinderDataset(val_tsv, resize=1024)

    weights = build_sampler_weights(train_dset, aug_weight=0.3)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, sampler=sampler,          
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Model: AnchorDETR (decode_head)
    cellsam = CellSAM({}).to(device).eval()
    sd = torch.load(args.init_cellsam_ckpt, map_location="cpu")
    cellsam.load_state_dict(sd, strict=False)

    for p in cellsam.model.parameters():
        p.requires_grad = False
    for p in cellsam.model_cp.parameters():
        p.requires_grad = False
    
    decode_head = cellsam.cellfinder.decode_head
    decode_head.train()
    
    freeze_N = max(int(args.freeze_backbone_epochs), 0)
    if freeze_N > 0:
        set_backbone_trainable(decode_head=decode_head, trainable=False)
        print(f"[INFO] Backbone frozen for first {freeze_N} epochs.")
    else:
        print("[INFO] Backbone not frozen.") 

    # Criterion
    class A: pass
    matcher_args = A()
    matcher_args.set_cost_class = 1.0
    matcher_args.set_cost_bbox  = 5.0
    matcher_args.set_cost_giou  = 2.0
    matcher_args.num_classes = 2
    
    matcher = build_matcher(matcher_args)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    weight_dict = add_aux_weights(weight_dict, num_dec_layers=6)
    losses = ["labels", "boxes"]
    criterion = SetCriterion(matcher_args.num_classes, matcher, weight_dict, losses).to(device)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in decode_head.named_parameters() if "backbone" not in n and p.requires_grad], "lr": args.lr},
        {"params": [p for n, p in decode_head.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optim = AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        
        if freeze_N > 0 and epoch == freeze_N + 1:
            set_backbone_trainable(decode_head=decode_head, trainable=True)
            print(f"[INFO] Backbone unfrozen at epoch {epoch}. Using lr_backbone={args.lr_backbone}")
            optim = AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        
        t0 = time.time()
        train_loss = train_one_epoch(cellfinder=decode_head, 
                                     cellsam=cellsam, 
                                     train_loader=train_loader, 
                                     criterion=criterion, 
                                     optimizer=optim, 
                                     device=device)
        val_loss = evaluate(cellfinder=decode_head, 
                            cellsam=cellsam, 
                            val_loader=val_loader, 
                            criterion=criterion, 
                            device=device)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.4f} val={val_loss:.4f}  (sec={dt/60:.1f})", flush=True)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decode_head.state_dict(), best_ckpt)
            print(f"Saved best -> {best_ckpt}", flush=True)

        # log + plot every epoch
        append_loss_row(loss_tsv, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val": best_val_loss})
        plot_losses(loss_tsv, loss_png)

    print("Training complete. Best checkpoint saved to:", best_ckpt)

if __name__ == "__main__":
    main()