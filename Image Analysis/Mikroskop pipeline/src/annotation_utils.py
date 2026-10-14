"""
Bridges hand-drawn napari/Fiji instance label masks to the bboxes TSV format
expected by 03_train_cellfinder.py and 04_train_decoder.py:
uid, image_path, label_path, xmin, ymin, xmax, ymax, label_id
"""
import os
import glob
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops


BOX_COLS = ["xmin", "ymin", "xmax", "ymax"]


def label_mask_to_rows(label_img: np.ndarray, uid: str, image_path: str, label_path: str,
                       keep_empty: bool = False) -> pd.DataFrame:
    """
    One row per instance in an instance-label mask (background=0, each
    object a unique positive integer id, e.g. what napari's Labels layer
    produces).

    keep_empty=True: a mask without instances (verified negative image, no
    gas vesicles) still yields ONE row with empty box columns (NaN), so the
    image stays in the TSV. Consumers drop NaN boxes per image -> 0 targets.
    """
    rows = []
    for region in regionprops(label_img):
        y0, x0, y1, x1 = region.bbox  # skimage bbox: (min_row, min_col, max_row, max_col)
        rows.append({
            "uid": uid,
            "image_path": image_path,
            "label_path": label_path,
            "xmin": x0,
            "ymin": y0,
            "xmax": x1,
            "ymax": y1,
            "label_id": region.label,
        })
    if not rows and keep_empty:
        rows.append({"uid": uid, "image_path": image_path, "label_path": label_path,
                     "xmin": np.nan, "ymin": np.nan, "xmax": np.nan, "ymax": np.nan,
                     "label_id": np.nan})
    return pd.DataFrame(rows)


def build_bboxes_csv(
    image_dir: str,
    mask_dir: str,
    out_csv: str,
    image_ext: str = ".tif",
    mask_suffix: str = "_mask",
) -> pd.DataFrame:
    """
    Match raw images in image_dir with instance-label masks in mask_dir
    (same filename stem + mask_suffix) and write the combined bboxes TSV.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, f"*{image_ext}")))
    all_rows = []

    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(mask_dir, f"{stem}{mask_suffix}{image_ext}")
        if not os.path.exists(label_path):
            print(f"[WARN] no mask found for {image_path}, skipping")
            continue

        label_img = tiff.imread(label_path)
        df = label_mask_to_rows(label_img, uid=stem, image_path=image_path, label_path=label_path)
        if df.empty:
            print(f"[WARN] mask for {stem} has no instances")
            continue
        all_rows.append(df)

    if not all_rows:
        raise ValueError(f"No matched image/mask pairs found in {image_dir} / {mask_dir}")

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_csv(out_csv, sep="\t", index=False)
    print(f"[INFO] wrote {len(out_df)} boxes across {out_df['uid'].nunique()} images -> {out_csv}")
    return out_df


def parse_args():
    import argparse
    ap = argparse.ArgumentParser("Build a bboxes TSV from raw images + instance-label masks.")
    ap.add_argument("--image_dir", required=True, help="e.g. data/01_raw")
    ap.add_argument("--mask_dir", required=True, help="e.g. data/masks_gvs or data/masks_cells")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--image_ext", default=".tif")
    ap.add_argument("--mask_suffix", default="_mask", help="mask filename = <image_stem><mask_suffix><image_ext>")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_bboxes_csv(args.image_dir, args.mask_dir, args.out_csv, args.image_ext, args.mask_suffix)
