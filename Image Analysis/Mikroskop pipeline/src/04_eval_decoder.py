#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

import torch
from bioio import BioImage

from cellSAM.sam_inference import CellSAM
from cellSAM.utils import normalize_image, format_image_shape
from cellSAM.model import get_local_model


# -------------------------
# IO
# -------------------------
def read_image_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff"]:
        arr = tiff.imread(path)
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            arr = arr.max(axis=0)
        if arr.ndim != 2:
            raise ValueError(f"TIFF not 2D after squeeze/MIP. shape={arr.shape} path={path}")
        return arr.astype(np.float32, copy=False)

    if ext == ".czi":
        img = BioImage(path)
        d = np.asarray(img.data)
        if d.ndim == 5:      # T C Z Y X
            zyx = d[0, 0]
        elif d.ndim == 4:    # C Z Y X
            zyx = d[0]
        elif d.ndim == 3:    # Z Y X
            zyx = d
        else:
            raise ValueError(f"Unsupported CZI ndim={d.ndim} shape={d.shape} path={path}")
        return zyx.max(axis=0).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported extension: {ext} for {path}")


def img2d_to_cellsam_tensor(img2d: np.ndarray) -> torch.Tensor:
    img = format_image_shape(img2d)                 # (H,W,3)
    img = normalize_image(img).astype(np.float32)   # (H,W,3)
    img = np.transpose(img, (2, 0, 1))              # (3,H,W)
    return torch.from_numpy(img)                    # CPU float32


# -------------------------
# Pixel FG/BG metrics
# -------------------------
def binarize_labels(lbl: np.ndarray) -> np.ndarray:
    return (lbl.astype(np.int64) > 0)


def update_confusion(tp_fp_fn, pred_fg: np.ndarray, gt_fg: np.ndarray):
    TP, FP, FN = tp_fp_fn
    TP += int(np.logical_and(pred_fg, gt_fg).sum())
    FP += int(np.logical_and(pred_fg, np.logical_not(gt_fg)).sum())
    FN += int(np.logical_and(np.logical_not(pred_fg), gt_fg).sum())
    return TP, FP, FN


def prf(TP: int, FP: int, FN: int):
    precision = TP / max(TP + FP, 1)
    recall    = TP / max(TP + FN, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


# -------------------------
# Instance IoU + matching
# -------------------------
def compact_relabel(lbl: np.ndarray) -> np.ndarray:
    """
    Relabel nonzero instance IDs compactly to 1..N.
    Keeps background 0.
    """
    lbl = lbl.astype(np.int64, copy=False)
    ids = np.unique(lbl)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros_like(lbl, dtype=np.uint16)

    remap = np.zeros(int(ids.max()) + 1, dtype=np.uint32)
    remap[ids] = np.arange(1, ids.size + 1, dtype=np.uint32)
    out = remap[lbl]
    return out.astype(np.uint16)


def instance_areas(lbl: np.ndarray) -> np.ndarray:
    """
    areas[id] = pixel count for instance id.
    Includes id=0 background.
    """
    return np.bincount(lbl.ravel().astype(np.int64))


def filter_ids_by_area(lbl: np.ndarray, area_min: int, area_max=None):
    """
    Return instance IDs satisfying area_min <= area <= area_max.
    If area_max is None, no upper bound is applied.
    """
    lbl = lbl.astype(np.int64, copy=False)
    areas = instance_areas(lbl)
    ids = np.arange(len(areas))
    ids = ids[ids != 0]

    keep = areas[ids] >= area_min
    if area_max is not None:
        keep &= areas[ids] <= area_max

    return ids[keep]


def relabel_keep_ids(lbl: np.ndarray, keep_ids: np.ndarray) -> np.ndarray:
    """
    Keep only selected instance IDs, set all others to 0,
    then compactly relabel to 1..N.
    """
    lbl = lbl.astype(np.int64, copy=False)
    if keep_ids.size == 0:
        return np.zeros_like(lbl, dtype=np.uint16)

    keep = np.zeros(int(lbl.max()) + 1, dtype=bool)
    keep[keep_ids] = True

    out = lbl.copy()
    out[~keep[out]] = 0
    return compact_relabel(out)


def instance_iou_and_match(gt: np.ndarray, pr: np.ndarray, iou_thr: float = 0.5):
    """
    gt, pr: label images with background=0.
    Returns TP, FP, FN, matched_ious.
    Matching is one-to-one greedy by IoU.
    """
    gt = compact_relabel(gt)
    pr = compact_relabel(pr)

    gt_max = int(gt.max())
    pr_max = int(pr.max())

    if gt_max == 0 and pr_max == 0:
        return 0, 0, 0, []
    if gt_max == 0:
        return 0, pr_max, 0, []
    if pr_max == 0:
        return 0, 0, gt_max, []

    gt_flat = gt.ravel().astype(np.int64)
    pr_flat = pr.ravel().astype(np.int64)

    K = gt_max + 1
    pair_ids = pr_flat * K + gt_flat
    inter = np.bincount(pair_ids, minlength=(pr_max + 1) * K).reshape((pr_max + 1, K)).astype(np.float64)

    area_pr = np.bincount(pr_flat, minlength=pr_max + 1).astype(np.float64)
    area_gt = np.bincount(gt_flat, minlength=gt_max + 1).astype(np.float64)

    inter_fg = inter[1:, 1:]
    area_pr_fg = area_pr[1:]
    area_gt_fg = area_gt[1:]

    union = area_pr_fg[:, None] + area_gt_fg[None, :] - inter_fg
    iou = inter_fg / np.maximum(union, 1e-9)  # shape: pred x gt

    pairs = []
    for pid in range(iou.shape[0]):
        good = np.where(iou[pid, :] >= iou_thr)[0]
        for gid in good:
            pairs.append((pid, gid, float(iou[pid, gid])))

    pairs.sort(key=lambda x: x[2], reverse=True)

    used_p = set()
    used_g = set()
    matched = []

    for pid, gid, val in pairs:
        if pid in used_p or gid in used_g:
            continue
        used_p.add(pid)
        used_g.add(gid)
        matched.append(val)

    tp = len(matched)
    fp = pr_max - tp
    fn = gt_max - tp
    return tp, fp, fn, matched


def eval_size_bucket(gt_lbl, pr_lbl, area_min: int, area_max, iou_thr: float):
    """
    Evaluate only instances in a specific size bucket.
    Both GT and predicted instances are filtered by their own area.
    """
    gt_ids = filter_ids_by_area(gt_lbl, area_min=area_min, area_max=area_max)
    pr_ids = filter_ids_by_area(pr_lbl, area_min=area_min, area_max=area_max)

    gt_f = relabel_keep_ids(gt_lbl, gt_ids)
    pr_f = relabel_keep_ids(pr_lbl, pr_ids)

    return instance_iou_and_match(gt_f, pr_f, iou_thr=iou_thr)


def empty_inst_stats():
    return {"TP": 0, "FP": 0, "FN": 0, "ious": []}


def update_inst_stats(stats, tp, fp, fn, ious):
    stats["TP"] += int(tp)
    stats["FP"] += int(fp)
    stats["FN"] += int(fn)
    stats["ious"].extend(ious)


def pq_sq_rq(tp: int, fp: int, fn: int, ious):
    """
    Panoptic-style metrics:
      PQ = sum IoU / (TP + 0.5FP + 0.5FN)
      SQ = sum IoU / TP
      RQ = TP / (TP + 0.5FP + 0.5FN)
    """
    denom = tp + 0.5 * fp + 0.5 * fn
    sum_iou = float(np.sum(ious)) if len(ious) else 0.0

    pq = sum_iou / denom if denom > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else 0.0
    rq = tp / denom if denom > 0 else 0.0

    return pq, sq, rq


def summarize_inst_stats(stats, prefix: str):
    tp = stats["TP"]
    fp = stats["FP"]
    fn = stats["FN"]
    ious = stats["ious"]

    precision, recall, f1 = prf(tp, fp, fn)
    mean_iou = float(np.mean(ious)) if len(ious) else 0.0
    pq, sq, rq = pq_sq_rq(tp, fp, fn, ious)

    return {
        f"{prefix}_TP": tp,
        f"{prefix}_FP": fp,
        f"{prefix}_FN": fn,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_mean_iou": mean_iou,
        f"{prefix}_PQ": pq,
        f"{prefix}_SQ": sq,
        f"{prefix}_RQ": rq,
    }


# Optional: drop predicted instances touching border (to match exclude_border GT)
def drop_border_instances(lbl: np.ndarray, margin: int = 0) -> np.ndarray:
    """
    Removes instances whose bbox touches image border within margin pixels.
    """
    
    if margin is None or margin <= 0:
        # still remove exact border-touching if margin==0? user controls; keep as-is
        return lbl

    H, W = lbl.shape
    out = lbl.copy()
    max_id = int(out.max())
    if max_id == 0:
        return out

    # find bbox for each id using min/max over coordinates (fast enough)
    # We'll do a single pass using where per id only if needed.
    # For many instances, this can be slower; keep margin usage optional.
    for pid in range(1, max_id + 1):
        ys, xs = np.where(out == pid)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        if (x0 <= margin) or (y0 <= margin) or (x1 >= (W - 1 - margin)) or (y1 >= (H - 1 - margin)):
            out[out == pid] = 0

    # relabel compactly to keep ids small (optional but nice)
    # map old ids -> new ids
    ids = np.unique(out)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.zeros_like(out, dtype=np.uint16)

    remap = {int(old): i+1 for i, old in enumerate(ids.tolist())}
    rel = np.zeros_like(out, dtype=np.uint16)
    for old, new in remap.items():
        rel[out == old] = new
        
    return rel


# -------------------------
# Model inference
# -------------------------
@torch.no_grad()
def predict_instance_labels(cellsam: CellSAM, img2d: np.ndarray, device: str, bbox_threshold: float):
    
    img = img2d_to_cellsam_tensor(img2d)  # (3,H,W)
    batch = img.float().unsqueeze(0).to(device)   # (1,3,H,W)

    cellsam.bbox_threshold = float(bbox_threshold)
    inst_labels, _masks, _emb, _boxes = cellsam.predict(batch, boxes_per_heatmap=None)

    if inst_labels is None:
        H, W = img2d.shape
        return np.zeros((H, W), dtype=np.uint16)
    
    return np.asarray(inst_labels).astype(np.uint16)


# -------------------------
# Main
# -------------------------
def parse_args():
    
    ap = argparse.ArgumentParser("Evaluate CellSAM masks: pixel FG/BG + instance IoU matching.")
    
    ap.add_argument("--outdir", required=True, help="Outdir containing test_bboxes.tsv and finetuned ckpt.")
    ap.add_argument("--old_ckpt",
                    default="/home/hd/hd_hd/hd_tt333/.deepcell/models/cellsam_v1.2/cellsam_general.pt",
                    help="Base CellSAM checkpoint (e.g. cellsam_general.pt).")
    ap.add_argument("--new_ckpt", default=None, help="Finetuned full CellSAM checkpoint. Default: outdir/decoder_ft3_ckpt.pt")
    ap.add_argument("--bbox_threshold", type=float, default=0.4, help="bbox_threshold for CellSAM.predict (use same for both).")
    ap.add_argument("--inst_iou_thr", type=float, default=0.5, help="IoU threshold for instance matching.")
    ap.add_argument("--save_new_preds", action="store_true", help="Save new model predicted instance labels for test set.")
    ap.add_argument("--pred_subdir", default="test_preds_new", help="Subfolder under outdir to save new predictions.")
    ap.add_argument("--border_margin", type=int, default=1, help="If >0, drop predicted instances touching border within this margin (pixels).")
    ap.add_argument("--micro_area_min", type=int, default=50, help="Minimum GT/pred instance area for micronuclei bucket.")
    ap.add_argument("--micro_area_max", type=int, default=300, help="Maximum GT/pred instance area for micronuclei bucket.")
    ap.add_argument("--micro_iou_thr", type=float, default=0.3, help="IoU threshold for micronuclei instance matching.")
    ap.add_argument("--nuc_area_min", type=int, default=301, help="Minimum GT/pred instance area for nuclei bucket.")
    ap.add_argument("--nuc_iou_thr", type=float, default=0.5, help="IoU threshold for nuclei instance matching.")
    
    return ap.parse_args()


def main():
    
    args = parse_args()
    if args.new_ckpt is None:
        args.new_ckpt = os.path.join(args.outdir, "decoder_ft3_ckpt.pt")
    print("[INFO] old_ckpt:", args.old_ckpt)
    print("[INFO] new_ckpt:", args.new_ckpt)

    test_tsv = os.path.join(args.outdir, "test_bboxes.tsv")

    df = pd.read_csv(test_tsv, sep="\t")
    groups = list(df.groupby("uid"))
    print(f"[INFO] Test images: {len(groups)}  (rows={len(df)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    old_model = get_local_model(args.old_ckpt)
    old_model.to(device).eval()
    
    new_model = get_local_model(args.new_ckpt)
    new_model.to(device).eval()
    
    save_dir = os.path.join(args.outdir, args.pred_subdir)
    if args.save_new_preds:
        os.makedirs(save_dir, exist_ok=True)
        print("[INFO] Saving new predictions to:", save_dir)

    # Pixel totals
    old_px = (0, 0, 0)  # TP, FP, FN
    new_px = (0, 0, 0)

    # Overall instance metrics
    old_inst_all = empty_inst_stats()
    new_inst_all = empty_inst_stats()

    # Size-stratified instance metrics
    old_micro = empty_inst_stats()  # area 50-300, IoU 0.3
    new_micro = empty_inst_stats()

    old_nuc = empty_inst_stats()    # area >300, IoU 0.5
    new_nuc = empty_inst_stats()

    for uid, g in tqdm(groups, desc="Eval"):
        img_path = g["image_path"].iloc[0]
        label_path = g["label_path"].iloc[0]

        img2d = read_image_any(img_path)
        gt_lbl = np.asarray(tiff.imread(label_path)).astype(np.uint16)
        gt_fg = binarize_labels(gt_lbl)

        # OLD prediction
        old_pred = predict_instance_labels(old_model, img2d, device=device, bbox_threshold=args.bbox_threshold)
        if args.border_margin > 0:
            old_pred = drop_border_instances(old_pred, margin=args.border_margin)

        old_fg = binarize_labels(old_pred)
        old_px = update_confusion(old_px, old_fg, gt_fg)

        # Overall instance metrics
        itp, ifp, ifn, ious = instance_iou_and_match(gt_lbl, old_pred, iou_thr=args.inst_iou_thr)
        update_inst_stats(old_inst_all, itp, ifp, ifn, ious)

        # Micronuclei bucket: area 50-300, IoU 0.3
        itp, ifp, ifn, ious = eval_size_bucket(
            gt_lbl, old_pred,
            area_min=args.micro_area_min,
            area_max=args.micro_area_max,
            iou_thr=args.micro_iou_thr,
        )
        update_inst_stats(old_micro, itp, ifp, ifn, ious)

        # Nuclei bucket: area >300, IoU 0.5
        itp, ifp, ifn, ious = eval_size_bucket(
            gt_lbl, old_pred,
            area_min=args.nuc_area_min,
            area_max=None,
            iou_thr=args.nuc_iou_thr,
        )
        update_inst_stats(old_nuc, itp, ifp, ifn, ious)

        # NEW prediction
        new_pred = predict_instance_labels(new_model, img2d, device=device, bbox_threshold=args.bbox_threshold)
        if args.border_margin > 0:
            new_pred = drop_border_instances(new_pred, margin=args.border_margin)

        new_fg = binarize_labels(new_pred)
        new_px = update_confusion(new_px, new_fg, gt_fg)

        # Overall instance metrics
        itp, ifp, ifn, ious = instance_iou_and_match(gt_lbl, new_pred, iou_thr=args.inst_iou_thr)
        update_inst_stats(new_inst_all, itp, ifp, ifn, ious)

        # Micronuclei bucket: area 50-300, IoU 0.3
        itp, ifp, ifn, ious = eval_size_bucket(
            gt_lbl, new_pred,
            area_min=args.micro_area_min,
            area_max=args.micro_area_max,
            iou_thr=args.micro_iou_thr,
        )
        update_inst_stats(new_micro, itp, ifp, ifn, ious)

        # Nuclei bucket: area >300, IoU 0.5
        itp, ifp, ifn, ious = eval_size_bucket(
            gt_lbl, new_pred,
            area_min=args.nuc_area_min,
            area_max=None,
            iou_thr=args.nuc_iou_thr,
        )
        update_inst_stats(new_nuc, itp, ifp, ifn, ious)

        if args.save_new_preds:
            out_path = os.path.join(save_dir, f"{uid}_pred_labels.tif")
            tiff.imwrite(out_path, new_pred, compression="zlib")

        # Pixel results
    old_p, old_r, old_f1 = prf(*old_px)
    new_p, new_r, new_f1 = prf(*new_px)

    old_all_summary = summarize_inst_stats(old_inst_all, "inst_all")
    new_all_summary = summarize_inst_stats(new_inst_all, "inst_all")

    old_micro_summary = summarize_inst_stats(old_micro, "micro")
    new_micro_summary = summarize_inst_stats(new_micro, "micro")

    old_nuc_summary = summarize_inst_stats(old_nuc, "nuc")
    new_nuc_summary = summarize_inst_stats(new_nuc, "nuc")

    old_balanced_f1 = 0.5 * old_micro_summary["micro_f1"] + 0.5 * old_nuc_summary["nuc_f1"]
    new_balanced_f1 = 0.5 * new_micro_summary["micro_f1"] + 0.5 * new_nuc_summary["nuc_f1"]

    results = [
        dict(
            name="old_base_cellsam",
            px_TP=old_px[0], px_FP=old_px[1], px_FN=old_px[2],
            px_precision=old_p, px_recall=old_r, px_f1=old_f1,
            balanced_micro_nuc_f1=old_balanced_f1,
            **old_all_summary,
            **old_micro_summary,
            **old_nuc_summary,
        ),
        dict(
            name="new_finetuned_cellsam",
            px_TP=new_px[0], px_FP=new_px[1], px_FN=new_px[2],
            px_precision=new_p, px_recall=new_r, px_f1=new_f1,
            balanced_micro_nuc_f1=new_balanced_f1,
            **new_all_summary,
            **new_micro_summary,
            **new_nuc_summary,
        )
    ]

    print("\nRESULTS")
    for r in results:
        print(r)

    out_tsv = os.path.join(args.outdir, "eval_results.tsv")
    pd.DataFrame(results).to_csv(out_tsv, sep="\t", index=False)
    print("\n[INFO] Wrote:", out_tsv)


if __name__ == "__main__":
    main()