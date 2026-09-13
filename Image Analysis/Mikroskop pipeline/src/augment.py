#!/usr/bin/env python3
"""
Augmentierung fuer annotierte Bilder + Instanz-Masken
=====================================================

Vorgehen (wichtig gegen Data Leakage):
  1. ZUERST die Originalbilder in train / val / test splitten.
  2. NUR den Train-Split augmentieren (Rotation/Spiegelung).
  3. val + test bleiben ausschliesslich Originale -> ehrliche Evaluation.

Der Advisor wollte "nicht zu viele" Rotationen -> Default sind nur 90 und 270
Grad (verlustfrei, keine Rand-Artefakte). Zusaetzlich werden augmentierte
Dateien im Namen mit "_aug" markiert (und damit auch in der uid), sodass ein
WeightedRandomSampler im Trainingsskript sie erkennen und seltener ziehen kann.

Ausgabe:
  <out_dir>/aug_images/           augmentierte Bilder (Originale bleiben, wo sie sind)
  <out_dir>/aug_masks/            augmentierte Masken
  <out_dir>/train_bboxes.tsv      Original-Train + Augmentierungen
  <out_dir>/val_bboxes.tsv        nur Originale
  <out_dir>/test_bboxes.tsv       nur Originale

Benutzung:
    python augment.py \\
        --image_dir data/01_raw \\
        --mask_dir data/masks_gvs \\
        --out_dir data_split \\
        --angles 90 270 \\
        --flips horizontal

Die drei TSVs kannst du direkt an die Trainings-/Eval-Skripte weiterreichen.
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import rotate as ndi_rotate

from annotation_utils import label_mask_to_rows


# -------------------------
# Transformationen
# -------------------------
def rotate_pair(image, mask, angle):
    """Bild interpoliert (order=3), Maske nearest-neighbor (order=0),
    damit die ganzzahligen Instanz-IDs nicht verfaelscht werden.
    reshape=False haelt die Bildgroesse konstant."""
    img_r = ndi_rotate(image, angle, reshape=False, order=3, mode="reflect").astype(image.dtype)
    msk_r = ndi_rotate(mask.astype(np.float32), angle, reshape=False, order=0,
                       mode="constant", cval=0).astype(mask.dtype)
    return img_r, msk_r


def flip_pair(image, mask, mode):
    if mode == "horizontal":
        return np.fliplr(image).copy(), np.fliplr(mask).copy()
    if mode == "vertical":
        return np.flipud(image).copy(), np.flipud(mask).copy()
    raise ValueError(f"Unbekannter flip mode: {mode}")


# -------------------------
# Split
# -------------------------
def split_stems(stems, val_frac, test_frac, seed):
    """Teilt die Liste der Bild-Stems in train/val/test (auf Bildebene)."""
    stems = list(stems)
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split-Anteile zu gross; Train-Split waere leer.")

    return stems[:n_train], stems[n_train:n_train + n_val], stems[n_train + n_val:]


# -------------------------
# Zeilen/TSV-Aufbau
# -------------------------
def rows_for_original(stem, image_path, mask_path):
    """bboxes-Zeilen fuer ein Originalbild (aus der Maske via regionprops)."""
    label_img = tiff.imread(mask_path)
    return label_mask_to_rows(label_img, uid=stem, image_path=image_path, label_path=mask_path)


def rows_for_augmented(image, mask, uid, out_image_path, out_mask_path):
    """Augmentiertes Bild+Maske auf Disk schreiben und bboxes-Zeilen erzeugen."""
    tiff.imwrite(out_image_path, image)
    tiff.imwrite(out_mask_path, mask, compression="zlib")
    return label_mask_to_rows(mask, uid=uid, image_path=out_image_path, label_path=out_mask_path)


def mask_n_instances(mask_path):
    """Anzahl annotierter Instanzen in einer Maske (0 = leere/nicht annotierte Maske)."""
    m = np.asarray(tiff.imread(mask_path))
    ids = np.unique(m)
    ids = ids[ids != 0]   # Hintergrund (0) ignorieren
    return int(ids.size)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Split + Augmentierung (nur Train) fuer Bild/Masken-Paare.")
    ap.add_argument("--image_dir", required=True, help="Ordner mit Original-Rohbildern (.tif)")
    ap.add_argument("--mask_dir", required=True, help="Ordner mit Original-Instanz-Masken (.tif)")
    ap.add_argument("--out_dir", required=True, help="Zielordner fuer augmentierte Dateien + die drei TSVs")
    ap.add_argument("--image_ext", default=".tif")
    ap.add_argument("--mask_suffix", default="_mask", help="Maske = <stem><mask_suffix><image_ext>")
    ap.add_argument("--angles", type=float, nargs="*", default=[90, 270],
                    help="Rotationswinkel fuer Train-Augmentierung. Leer = keine Rotation.")
    ap.add_argument("--flips", nargs="*", default=[], choices=["horizontal", "vertical"],
                    help="Spiegelungen fuer Train-Augmentierung. Leer = keine.")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    aug_image_dir = os.path.join(args.out_dir, "aug_images")
    aug_mask_dir = os.path.join(args.out_dir, "aug_masks")
    os.makedirs(aug_image_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)

    # 1) Bild/Masken-Paare finden
    pairs = {}  # stem -> (image_path, mask_path)
    for image_path in sorted(glob.glob(os.path.join(args.image_dir, f"*{args.image_ext}"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(args.mask_dir, f"{stem}{args.mask_suffix}{args.image_ext}")
        if not os.path.exists(mask_path):
            print(f"[WARN] keine Maske fuer {stem}, ueberspringe.")
            continue
        pairs[stem] = (image_path, mask_path)

    if not pairs:
        raise ValueError("Keine passenden Bild/Masken-Paare gefunden.")
    print(f"[INFO] {len(pairs)} Original-Paare gefunden.")

    # 1b) Leere (nicht annotierte) Masken erkennen und aussortieren
    nonempty = {}
    empty_stems = []
    for stem, (image_path, mask_path) in pairs.items():
        if mask_n_instances(mask_path) > 0:
            nonempty[stem] = (image_path, mask_path)
        else:
            empty_stems.append(stem)

    if empty_stems:
        print(f"[WARN] {len(empty_stems)} Masken sind LEER (keine Annotation) und werden ausgeschlossen:")
        for s in empty_stems:
            print(f"       - {s}")

    print(f"[INFO] {len(nonempty)} Bilder mit Annotation verbleiben.")
    if len(nonempty) < 3:
        raise ValueError("Zu wenige annotierte Bilder fuer einen train/val/test-Split. "
                         "Bitte mehr Masken annotieren.")

    # 2) ZUERST splitten (nur ueber annotierte Bilder -> kein leeres Test-Set)
    pairs = nonempty
    train_stems, val_stems, test_stems = split_stems(
        pairs.keys(), args.val_frac, args.test_frac, args.seed
    )
    print(f"[INFO] Split: train={len(train_stems)}  val={len(val_stems)}  test={len(test_stems)}")

    # 3) val + test: nur Originale
    val_rows = [rows_for_original(s, *pairs[s]) for s in val_stems]
    test_rows = [rows_for_original(s, *pairs[s]) for s in test_stems]

    # 4) train: Originale + Augmentierungen (nur hier!)
    train_rows = []
    for stem in train_stems:
        image_path, mask_path = pairs[stem]

        # Original
        train_rows.append(rows_for_original(stem, image_path, mask_path))

        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)

        # Rotationen
        for angle in args.angles:
            img_r, msk_r = rotate_pair(image, mask, angle)
            uid = f"{stem}_aug_rot{int(angle)}"
            out_img = os.path.join(aug_image_dir, f"{uid}{args.image_ext}")
            out_msk = os.path.join(aug_mask_dir, f"{uid}{args.mask_suffix}{args.image_ext}")
            train_rows.append(rows_for_augmented(img_r, msk_r, uid, out_img, out_msk))

        # Spiegelungen
        for mode in args.flips:
            img_f, msk_f = flip_pair(image, mask, mode)
            uid = f"{stem}_aug_flip{mode[:1]}"
            out_img = os.path.join(aug_image_dir, f"{uid}{args.image_ext}")
            out_msk = os.path.join(aug_mask_dir, f"{uid}{args.mask_suffix}{args.image_ext}")
            train_rows.append(rows_for_augmented(img_f, msk_f, uid, out_img, out_msk))

    # 5) TSVs schreiben
    def write_tsv(rows, name):
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        path = os.path.join(args.out_dir, name)
        df.to_csv(path, sep="\t", index=False)
        n_imgs = df["uid"].nunique() if not df.empty else 0
        print(f"[INFO] {name}: {len(df)} Boxen ueber {n_imgs} Bilder -> {path}")

    write_tsv(train_rows, "train_bboxes.tsv")
    write_tsv(val_rows, "val_bboxes.tsv")
    write_tsv(test_rows, "test_bboxes.tsv")

    print("\n[DONE] Augmentierte Dateien tragen '_aug' in der uid.")
    print("       -> im Training per WeightedRandomSampler seltener ziehen.")


if __name__ == "__main__":
    main()
