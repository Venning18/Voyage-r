#!/usr/bin/env python3
import os
import argparse
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler

from bioio import BioImage
from cellSAM.sam_inference import CellSAM
from cellSAM.utils import normalize_image, format_image_shape


# -------------------------
# UTILS
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


def img2d_to_cellsam_tensor(img2d: np.ndarray) -> torch.Tensor:
    """
    Convert 2D image -> torch tensor (3,H,W) float32 in [0,1] using cellSAM.utils.normalize_image.
    """
    img = format_image_shape(img2d)             # (H,W,3)
    img = normalize_image(img).astype(np.float32)  # (H,W,3), ~[0,1]
    img = np.transpose(img, (2, 0, 1))          # (3,H,W)
    return torch.from_numpy(img)                # float32 on CPU

def split_by_uid(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    uids = df["uid"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(uids)

    n = len(uids)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split fractions too large; train split would be empty.")

    train_u = set(uids[:n_train])
    val_u   = set(uids[n_train:n_train + n_val])
    test_u  = set(uids[n_train + n_val:])

    train_df = df[df["uid"].isin(train_u)].copy()
    val_df   = df[df["uid"].isin(val_u)].copy()
    test_df  = df[df["uid"].isin(test_u)].copy()
    
    return train_df, val_df, test_df


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits/target: (H,W) float
    """
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def scale_boxes(boxes_xyxy: np.ndarray, W: int, H: int) -> np.ndarray:
    """
    boxes_xyxy in original pixel space -> 1024 pixel space (still xyxy), like CellSAM.predict scaling.
    """
    scale_x = 1024.0 / float(W)
    scale_y = 1024.0 / float(H)
    b = boxes_xyxy.astype(np.float32).copy()
    b[:, [0, 2]] *= scale_x
    b[:, [1, 3]] *= scale_y
    return b

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
    
def load_losses_state(loss_tsv: str) -> Tuple[int, float]:
    """
    Returns (next_epoch, best_val) from existing losses.tsv.
    If file missing/empty, returns (1, inf).
    """
    if not os.path.exists(loss_tsv):
        return 1, float("inf")
    df = pd.read_csv(loss_tsv, sep="\t")
    if df.empty or "epoch" not in df.columns:
        return 1, float("inf")
    last_epoch = int(df["epoch"].max())

    if "best_val" in df.columns and not df["best_val"].isna().all():
        best_val = float(df["best_val"].dropna().iloc[-1])
    else:
        best_val = float(df["val_loss"].min()) if "val_loss" in df.columns else float("inf")
    return last_epoch + 1, best_val

# -------------------------
# Dataset
# -------------------------
@dataclass
class Sample:
    uid: str
    image_path: str
    label_path: str
    img: torch.Tensor          # (3,H,W) float32 on CPU
    label_img: np.ndarray      # (H,W) uint16 on CPU
    boxes_xyxy: np.ndarray     # (N,4) float32 in original pixels
    label_ids: np.ndarray      # (N,) int


class MaskTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_instances_per_image: int = 16, seed: int = 0):
        self.groups = list(df.groupby("uid"))
        self.max_instances = int(max_instances_per_image)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx) -> Sample:
        uid, g = self.groups[idx]
        image_path = g["image_path"].iloc[0]
        label_path = g["label_path"].iloc[0]

        img2d = read_image_any(image_path)
        img = img2d_to_cellsam_tensor(img2d)  # (3,H,W) float32

        label_img = tiff.imread(label_path)
        label_img = np.asarray(label_img)

        # pick subset of instances (to control compute)
        rows = g[["xmin", "ymin", "xmax", "ymax", "label_id"]].dropna().copy()
        rows["label_id"] = rows["label_id"].astype(int)
        if len(rows) == 0:
            # degenerate, but keep shape consistent
            boxes = np.zeros((0, 4), dtype=np.float32)
            lids = np.zeros((0,), dtype=int)
        else:
            if self.max_instances > 0 and len(rows) > self.max_instances:
                rows = rows.sample(n=self.max_instances, random_state=self.rng.randint(0, 10**9))
            boxes = rows[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)
            lids = rows["label_id"].to_numpy(int)

        return Sample(
            uid=str(uid),
            image_path=str(image_path),
            label_path=str(label_path),
            img=img,
            label_img=label_img,
            boxes_xyxy=boxes,
            label_ids=lids,
        )


def collate_samples(batch: List[Sample]) -> List[Sample]:
    return batch

# -------------------------
# SAM model
# -------------------------
def forward_masks_for_boxes(
    cellsam: CellSAM,
    img_tensor: torch.Tensor,         # (3,H,W) float32 CPU
    boxes_xyxy: np.ndarray,           # (N,4) float32 original pixels
    paddings: Tuple[int, int],        # (padh, padw) from cellsam.prep_2
    image_embedding: torch.Tensor,    # (C, h, w) on device
) -> List[torch.Tensor]:
    
    """
    Returns list of mask logits, each (H,W) on device, aligned to original image size
    using SAM postprocess_masks
    """
    
    device = image_embedding.device
    mdl = cellsam.model_cp

    H, W = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])
    if boxes_xyxy.shape[0] == 0:
        return []

    boxes_1024 = scale_boxes(boxes_xyxy, W=W, H=H)
    out_masks: List[torch.Tensor] = []

    # input_size expects (H_in, W_in) before padding, see CellSAM.predict usage.
    input_size = torch.tensor([1024 - paddings[0], 1024 - paddings[1]], device=device)

    for b in boxes_1024:
        # shape (1,1,4)
        input_box = torch.as_tensor(b, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        sparse_embeddings, dense_embeddings = mdl.prompt_encoder(
            points=None,
            boxes=input_box,
            masks=None,
        )

        low_res_masks, _iou_pred = mdl.mask_decoder(
            image_embeddings=image_embedding.unsqueeze(0),
            image_pe=mdl.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # low_res_masks: (1,1,256,256) logits

        # upscale to original size (1,1,H,W)
        up = mdl.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=(H, W),
        )
        out_masks.append(up[0, 0])  # (H,W) logits

    return out_masks

def set_trainable_mask_modules(cellsam: CellSAM, train_neck: bool = True):
    
    # Freeze everything by default
    for p in cellsam.parameters():
        p.requires_grad = False

    for p in cellsam.cellfinder.parameters():
        p.requires_grad = False
        
    for p in cellsam.model_cp.mask_decoder.parameters():
        p.requires_grad = True
            
    if train_neck is True and hasattr(cellsam.model_cp.image_encoder, "neck"):
        for p in cellsam.model_cp.image_encoder.neck.parameters():
            p.requires_grad = True


def make_optimizer(cellsam: CellSAM, lr_decoder: float, lr_neck: float, weight_decay: float) -> AdamW:
    
    groups = []
    # decoder
    dec_params = [p for p in cellsam.model_cp.mask_decoder.parameters() if p.requires_grad]
    if dec_params:
        groups.append({"params": dec_params, "lr": lr_decoder})

    # neck (optional)
    if hasattr(cellsam.model_cp.image_encoder, "neck"):
        neck_params = [p for p in cellsam.model_cp.image_encoder.neck.parameters() if p.requires_grad]
        if neck_params:
            groups.append({"params": neck_params, "lr": lr_neck})

    if not groups:
        raise RuntimeError("No trainable parameters found. Did you freeze everything?")
    
    return AdamW(groups, weight_decay=weight_decay)


def load_decode_head_into_cellsam(cellsam: CellSAM, new_ckpt_path: str):

    sd = torch.load(os.path.expanduser(new_ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # strip possible prefixes
    if any(k.startswith("cellfinder.decode_head.") for k in sd.keys()):
        stripped = {k[len("cellfinder.decode_head."):]: v for k, v in sd.items() if k.startswith("cellfinder.decode_head.")}
        sd = stripped
    elif any(k.startswith("module.cellfinder.decode_head.") for k in sd.keys()):
        stripped = {k[len("module.cellfinder.decode_head."):]: v for k, v in sd.items() if k.startswith("module.cellfinder.decode_head.")}
        sd = stripped

    missing, unexpected = cellsam.cellfinder.decode_head.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded decode_head: missing={missing} unexpected={unexpected}")
    
    
# -------------------------
# Train / Eval
# -------------------------

def compute_batch_loss(cellsam: CellSAM, batch: List[Sample], device: str, bce_w: float, dice_w: float) -> torch.Tensor:
    """
    Forward one batch and return scalar loss (averaged over all instances in batch).
    """
    # 1) prep images with CellSAM preprocessing used for embeddings (prep_2 + forward)
    imgs = [s.img for s in batch]  # list of (3,H,W) float32 CPU
    transformed, paddings = cellsam.prep_2(imgs, percentile=True)  # tensor (B,3,1024,1024) + paddings
    transformed = transformed.to(device, non_blocking=True)

    # 2) image embeddings (B, ...)
    embeddings = cellsam.forward(transformed, return_preprocessed=False)

    total = torch.zeros((), device=device)
    n_inst = 0

    for i, s in enumerate(batch):
        if s.boxes_xyxy.shape[0] == 0:
            continue

        masks_logits = forward_masks_for_boxes(
            cellsam=cellsam,
            img_tensor=s.img,
            boxes_xyxy=s.boxes_xyxy,
            paddings=paddings[i],
            image_embedding=embeddings[i],
        )

        # GT masks from instance label image
        for lid, pred_logit in zip(s.label_ids.tolist(), masks_logits):
            gt = (s.label_img == int(lid)).astype(np.float32)  # (H,W)
            gt_t = torch.from_numpy(gt).to(device)

            bce = F.binary_cross_entropy_with_logits(pred_logit, gt_t)
            dsc = dice_loss_with_logits(pred_logit, gt_t)
            total = total + (bce_w * bce + dice_w * dsc)
            n_inst += 1

    if n_inst == 0:
        # avoid NaN if batch has no instances
        return total

    return total / float(n_inst)

@torch.no_grad()
def evaluate_epoch(cellsam: CellSAM, loader: DataLoader, device: str, bce_w: float, dice_w: float, use_amp:bool=True) -> float:
    
    cellsam.eval()
    running = 0.0
    n = 0
    
    for batch in tqdm(loader, desc="Val", leave=False):
        
        with torch.amp.autocast("cuda", enabled=(use_amp and device.startswith("cuda"))):
            loss = compute_batch_loss(cellsam, batch, device, bce_w, dice_w)
        running += float(loss.item())
        n += 1
        
    return running / max(n, 1)


def train_epoch(cellsam: CellSAM, loader: DataLoader, optim: AdamW, device: str, bce_w: float, dice_w: float, use_amp: bool = True) -> float:
    
    cellsam.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.startswith("cuda")))
    running = 0.0
    n = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.startswith("cuda"))):
            loss = compute_batch_loss(cellsam, batch, device, bce_w, dice_w)

        scaler.scale(loss).backward()
        scaler.unscale_(optim) 
        torch.nn.utils.clip_grad_norm_([p for p in cellsam.parameters() if p.requires_grad], 0.1)
        
        scaler.step(optim)
        scaler.update()

        running += float(loss.item())
        n += 1

    return running / max(n, 1)

# -------------------------
# Prediction writing
# -------------------------
@torch.no_grad()
def write_predictions(
    cellsam: CellSAM,
    df: pd.DataFrame,
    outdir: str,
    device: str,
    bbox_threshold: float,
):
    groups = list(df.groupby("uid"))
    cellsam.eval()
    cellsam.bbox_threshold = float(bbox_threshold)

    for uid, g in tqdm(groups, desc="Write preds"):
        img_path = g["image_path"].iloc[0]
        img2d = read_image_any(img_path)
        img = img2d_to_cellsam_tensor(img2d)  # (3,H,W) float

        # CellSAM.predict expects a batch-like tensor: (B,3,H,W)
        batch = img.unsqueeze(0).to(device)

        inst_labels, _masks, _emb, _boxes = cellsam.predict(batch, boxes_per_heatmap=None)
        # inst_labels is a numpy label map per code path in predict

        if inst_labels is None:
            pred = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint16)
        else:
            pred = np.asarray(inst_labels).astype(np.uint16)

        out_path = os.path.join(outdir, f"{uid}_pred_labels.tif")
        tiff.imwrite(out_path, pred, compression="zlib")
  
# -------------------------
# Main
# -------------------------
def parse_args():
    
    ap = argparse.ArgumentParser("Train CellSAM SAM mask decoder (stage-2) using GT labels + boxes.")
    
    ap.add_argument("--bboxes_csv", required=False, default=None, help="TSV with columns incl uid,image_path,label_path,xmin,ymin,xmax,ymax,label_id")
    ap.add_argument("--init_cellsam_ckpt",
                    default="/home/hd/hd_hd/hd_tt333/.deepcell/models/cellsam_v1.2/cellsam_general.pt",
                    help="Base CellSAM checkpoint (e.g. cellsam_general.pt)")
    ap.add_argument("--cellfinder_decode_head_ckpt", required=True, help="Your fine-tuned decode_head .pt")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_instances_per_image", type=int, default=200, help="How many GT instances to train on per image per epoch step")

    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2)

    ap.add_argument("--lr_decoder", type=float, default=1e-5)
    ap.add_argument("--lr_neck", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--bce_w", type=float, default=1.0)
    ap.add_argument("--dice_w", type=float, default=1.0)

    ap.add_argument("--use_existing_splits", action="store_true",
                    help="Nutze vorhandene TSVs statt intern zu splitten.")
    ap.add_argument("--split_dir", default=None, help="Ordner mit den TSVs. Default: --outdir.")

    ap.add_argument("--bbox_threshold_pred", type=float, default=0.4, help="bbox_threshold for writing final preds")
    ap.add_argument("--write_val_preds", action="store_true", help="After training, write instance-label predictions for VAL split")
    
    ap.add_argument("--resume", action="store_true", help="Resume training from checkpoint and existing train/val/test TSVs in outdir.")

    return ap.parse_args()


def main():
    
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    
    best_ckpt = os.path.join(args.outdir, "decoder_ft3_ckpt.pt")
    loss_tsv = os.path.join(args.outdir, "losses.tsv")
    loss_png = os.path.join(args.outdir, "loss_curve.png")

    if args.resume or args.use_existing_splits:
        split_dir = args.split_dir or args.outdir
        train_df = pd.read_csv(os.path.join(split_dir, "train_bboxes.tsv"), sep="\t")
        val_df   = pd.read_csv(os.path.join(split_dir, "val_bboxes.tsv"), sep="\t")
        test_df  = pd.read_csv(os.path.join(split_dir, "test_bboxes.tsv"), sep="\t")
        print(f"[INFO] Nutze vorhandene Splits aus: {split_dir}")

    else:
        # load data
        df = pd.read_csv(args.bboxes_csv, sep="\t")
        df = df.dropna(subset=["uid", "image_path", "label_path", "xmin", "ymin", "xmax", "ymax", "label_id"]).copy()
        train_df, val_df, test_df = split_by_uid(df, args.val_frac, args.test_frac, args.seed)

        train_df.to_csv(os.path.join(args.outdir, "train_bboxes.tsv"), sep="\t", index=False)
        val_df.to_csv(os.path.join(args.outdir, "val_bboxes.tsv"), sep="\t", index=False)
        test_df.to_csv(os.path.join(args.outdir, "test_bboxes.tsv"), sep="\t", index=False)
        print(f"[INFO] train images={train_df['uid'].nunique()} rows={len(train_df)}")
        print(f"[INFO] val   images={val_df['uid'].nunique()} rows={len(val_df)}")
        print(f"[INFO] test  images={test_df['uid'].nunique()} rows={len(test_df)}")

    # Negativbeispiele (Zeilen ohne Box) entfernen: der Decoder lernt Masken zu Boxen,
    # ohne Box gibt es nichts zu lernen (und ein Batch nur aus Negativen haette keinen Gradienten)
    box_cols = ["xmin", "ymin", "xmax", "ymax", "label_id"]
    train_df = train_df.dropna(subset=box_cols).copy()
    val_df = val_df.dropna(subset=box_cols).copy()

    train_set = MaskTrainDataset(train_df, max_instances_per_image=args.max_instances_per_image, seed=args.seed)
    val_set = MaskTrainDataset(val_df, max_instances_per_image=args.max_instances_per_image, seed=args.seed + 1)

    weights = build_sampler_weights(train_set, aug_weight=0.3)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=sampler,                 # statt shuffle=True
        num_workers=args.num_workers, collate_fn=collate_samples,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_samples,
        pin_memory=True,
    )

    # build model
    cellsam = CellSAM({}).to(device)
    
    base_sd = torch.load(os.path.expanduser(args.init_cellsam_ckpt), map_location="cpu")
    cellsam.load_state_dict(base_sd, strict=False)
    load_decode_head_into_cellsam(cellsam, args.cellfinder_decode_head_ckpt) # finetuned cellfinder decode_head
    cellsam.adv_mode = True # just to be sure, since CellSAM.from_pretrained sets this to True for SAM pretraining compatibility
    
    set_trainable_mask_modules(cellsam, train_neck=True) # train neck and decoder only

    # optimizer
    optim = make_optimizer(cellsam, args.lr_decoder, args.lr_neck, args.weight_decay)
    
    best_val = float("inf")
    start_epoch = 1

    if args.resume:
        start_epoch, best_val = load_losses_state(loss_tsv)
        sd = torch.load(os.path.expanduser(best_ckpt), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        cellsam.load_state_dict(sd, strict=False)

    # train
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(cellsam=cellsam, loader=train_loader, optim=optim, device=device, bce_w=args.bce_w, dice_w=args.dice_w)
        val_loss = evaluate_epoch(cellsam=cellsam, loader=val_loader, device=device, bce_w=args.bce_w, dice_w=args.dice_w)
        dt = time.time() - t0

        print(f"[E{epoch:03d}] train={train_loss:.5f} val={val_loss:.5f}  ({dt/60.0:.1f} min)", flush=True)

        # save best ckpt
        if val_loss < best_val:
            best_val = val_loss
            torch.save(cellsam.state_dict(), best_ckpt)
            print(f"[INFO] saved best -> {best_ckpt} (val={best_val:.5f})", flush=True)
            
        append_loss_row(loss_tsv, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val": best_val})
        plot_losses(loss_tsv, loss_png)

    print("[INFO] Training done.")
    print("[INFO] Best ckpt:", best_ckpt)

    if args.write_val_preds:
        preds_dir = os.path.join(args.outdir, "val_preds")
        print("[INFO] Writing val predictions to:", preds_dir)
        write_predictions(
            cellsam=cellsam,
            df=val_df,
            outdir=preds_dir,
            device=device,
            bbox_threshold=args.bbox_threshold_pred,
        )
        print("[INFO] Done writing preds.")


if __name__ == "__main__":
    main()