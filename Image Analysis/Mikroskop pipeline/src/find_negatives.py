#!/usr/bin/env python3
"""
Findet leere Masken (keine Gasvesikel) und schreibt die Bild-Stems in negatives.txt,
passend fuer augment.py --negatives.

ACHTUNG: Nur benutzen, wenn ALLE leeren Masken wirklich "keine GVs" bedeuten
(und nicht "noch nicht annotiert").

Benutzung:
    uv run python src/find_negatives.py --mask_dir data/masks_gvs --out data/negatives.txt
"""
import os
import glob
import argparse

import numpy as np
import tifffile as tiff


def main():
    ap = argparse.ArgumentParser(description="Leere Masken -> negatives.txt")
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out", default="negatives.txt")
    ap.add_argument("--mask_suffix", default="_mask")
    ap.add_argument("--image_ext", default=".tif")
    args = ap.parse_args()

    tail = f"{args.mask_suffix}{args.image_ext}"
    all_tifs = sorted(glob.glob(os.path.join(args.mask_dir, f"*{args.image_ext}")))
    masks = [p for p in all_tifs if p.endswith(tail)]

    # Dateien, die nicht auf <stem>_mask.tif enden, findet augment.py nicht
    for p in all_tifs:
        if not p.endswith(tail):
            print(f"[WARN] Name passt nicht zu '*{tail}', wird ignoriert: {os.path.basename(p)}")

    empty = []
    for p in masks:
        if not np.any(tiff.imread(p)):
            empty.append(os.path.basename(p)[: -len(tail)])

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Bilder OHNE Gasvesikel (leere Maske) -> augment.py --negatives\n")
        f.writelines(s + "\n" for s in empty)

    print(f"[INFO] {len(masks)} Masken geprueft, {len(empty)} leer -> {args.out}")


if __name__ == "__main__":
    main()
