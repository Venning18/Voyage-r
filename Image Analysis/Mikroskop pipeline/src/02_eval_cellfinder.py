import os
import random
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from bioio import BioImage
from tqdm import tqdm

import torch
import torch.nn.functional as F

from cellSAM.sam_inference import CellSAM

# ---------- IO ----------
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

def cxcywh_to_xyxy(boxes):
    
    # boxes: (N,4) normalized cxcywh -> normalized xyxy
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = cx - 0.5*w
    y1 = cy - 0.5*h
    x2 = cx + 0.5*w
    y2 = cy + 0.5*h
    xyxy = np.stack([x1,y1,x2,y2], axis=1)
    
    return np.clip(xyxy, 0.0, 1.0)

def filter_border_boxes(boxes_xyxy: np.ndarray, H: int, W: int, margin: int = 0) -> np.ndarray:
    """
    Remove boxes that touch the image border (within margin pixels).
    boxes_xyxy are in pixel coordinates.
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return boxes_xyxy

    b = boxes_xyxy.astype(np.float32, copy=True)

    # clip to image bounds first (protect against tiny numerical overshoot)
    b[:, 0] = np.clip(b[:, 0], 0, W - 1)
    b[:, 2] = np.clip(b[:, 2], 0, W - 1)
    b[:, 1] = np.clip(b[:, 1], 0, H - 1)
    b[:, 3] = np.clip(b[:, 3], 0, H - 1)

    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    # keep only boxes fully inside the margin
    keep = (x1 > margin) & (y1 > margin) & (x2 < (W - 1 - margin)) & (y2 < (H - 1 - margin))
    return b[keep]

def mip_to_uint8_rgb(img2d: np.ndarray, p_low=1.0, p_high=99.0) -> torch.Tensor:
    
    x = img2d.astype(np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    u8 = (x * 255.0).astype(np.uint8)
    rgb = np.stack([u8, u8, u8], axis=0)  # (3,H,W)
    
    return torch.from_numpy(rgb)


# ---------- Evaluation Metrics ----------
def iou_matrix_xyxy(a, b):
    
    # a: (N,4), b: (M,4) in normalized xyxy
    if len(a)==0 or len(b)==0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]
    
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)
    
    inter = iw * ih
    area_a = np.maximum(ax2-ax1, 0.0) * np.maximum(ay2-ay1, 0.0)
    area_b = np.maximum(bx2-bx1, 0.0) * np.maximum(by2-by1, 0.0)
    union = area_a + area_b - inter
    
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)

def greedy_match(iou, thr):
    
    # iou: (Npred, Ngt)
    Np, Ng = iou.shape
    matched_iou = []
    used_p = set()
    used_g = set()
    # sort all pairs by IoU desc
    pairs = [(i, j, iou[i,j]) for i in range(Np) for j in range(Ng)]
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, v in pairs:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched_iou.append(v)
    tp = len(matched_iou)
    fp = Np - tp
    fn = Ng - tp
    
    return tp, fp, fn, matched_iou

# ---------- Model build/load ----------
def load_cellsam(ckpt_path: str, device: str) -> CellSAM:
    
    model = CellSAM({}).to(device).eval()
    sd = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    
    return model


def load_decode_head_into_cellsam(cellsam: CellSAM, new_ckpt_path: str):
    
    sd = torch.load(new_ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
        
    if any(k.startswith("cellfinder.decode_head.") for k in sd.keys()):
        stripped = {}
        for k, v in sd.items():
            if k.startswith("cellfinder.decode_head."):
                stripped[k[len("cellfinder.decode_head."):]] = v
        sd = stripped
    elif any(k.startswith("module.cellfinder.decode_head.") for k in sd.keys()):
        stripped = {}
        for k, v in sd.items():
            if k.startswith("module.cellfinder.decode_head."):
                stripped[k[len("module.cellfinder.decode_head."):]] = v
        sd = stripped

    missing, unexpected = cellsam.cellfinder.decode_head.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded new decode_head: missing={missing} unexpected={unexpected}")

# ---------- Inference ----------
@torch.no_grad()
def predict_boxes_cellsam(cellsam: CellSAM, img_raw: np.ndarray, score_thr: float):
    
    H, W = img_raw.shape
    img = mip_to_uint8_rgb(img_raw)  # CPU tensor uint8 (3,H,W)

    # CellSAM preprocessing
    x = cellsam.sam_bbox_preprocessing([img], percentile=False)  # (1,3,1024,1024) float on device

    # forward_inference returns boxes in the target_size space (here 1024x1024)
    res = cellsam.cellfinder.forward_inference(x)[0]
    boxes_1024 = res["boxes"].detach().cpu().numpy().astype(np.float32)   # (N,4) xyxy in 1024 pixels
    scores = res["scores"].detach().cpu().numpy().astype(np.float32)

    keep = scores >= score_thr
    boxes_1024 = boxes_1024[keep]
    scores = scores[keep]

    # scale to original image pixel space
    if len(boxes_1024) > 0:
        boxes = boxes_1024.copy()
        boxes[:, [0, 2]] *= (W / 1024.0)
        boxes[:, [1, 3]] *= (H / 1024.0)
    else:
        boxes = boxes_1024

    return boxes, scores

# ---------- Main ----------
def parse_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--bboxes_test", required=True)
    ap.add_argument("--new_ckpt", required=True)
    ap.add_argument("--old_cellsam_ckpt",
                    default="/home/hd/hd_hd/hd_tt333/.deepcell/models/cellsam_v1.2/cellsam_general.pt",
                    help="Base CellSAM checkpoint.")
    ap.add_argument("--score_thr", type=float, default=0.4)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--border_margin", type=int, default=1)
    args = ap.parse_args()
    
    return args

def main():

    args = parse_args()
    
    df = pd.read_csv(args.bboxes_test, sep="\t")
    groups = list(df.groupby("uid"))
    print(f"[INFO] Test images: {len(groups)}  (bboxes={len(df)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build old/new models
    old_cellsam = load_cellsam(args.old_cellsam_ckpt, device=device)

    new_cellsam = load_cellsam(args.old_cellsam_ckpt, device=device)
    load_decode_head_into_cellsam(new_cellsam, args.new_ckpt)

    def eval_model(model, name):
        
        TP=FP=FN=0
        matched_ious=[]
        
        for uid, g in tqdm(groups, desc=f"Eval {name}"):
            img_path = g["image_path"].iloc[0]
            img = read_image_any(img_path)

            H, W = img.shape
            # Negativbeispiele: Zeile ohne Box -> 0 GT-Boxen, jede Vorhersage zaehlt als FP
            gt_xyxy = g[["xmin","ymin","xmax","ymax"]].dropna().to_numpy(np.float32)
            pred, scores = predict_boxes_cellsam(cellsam=model, img_raw=img, score_thr=args.score_thr)
            pred = filter_border_boxes(pred, H=H, W=W, margin=args.border_margin)
            
            iou = iou_matrix_xyxy(pred, gt_xyxy)
            tp, fp, fn, miou = greedy_match(iou, args.iou_thr)
            TP += tp; FP += fp; FN += fn
            matched_ious.extend(miou)

        prec = TP / max(TP+FP, 1)
        rec  = TP / max(TP+FN, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-9)
        miou = float(np.mean(matched_ious)) if matched_ious else 0.0
        
        return dict(name=name, TP=TP, FP=FP, FN=FN, precision=prec, recall=rec, f1=f1, mean_iou=miou)

    r_old = eval_model(old_cellsam, "old_cellsam_cellfinder")
    r_new = eval_model(new_cellsam, "new_finetuned_cellfinder")

    print("\nRESULTS")
    for r in [r_old, r_new]:
        print(r)

if __name__ == "__main__":
    main()