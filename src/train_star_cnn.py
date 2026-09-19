#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Star RGB Multitask CNN — staged training, combined subclass head, confusion matrices, and training curves.

Outputs:
- Trained Keras model: best_star_model_staged_combined.keras + final .keras
- Staged training history: training_history_staged_combined.json/.csv
- Confusion matrices (letters, subclass-per-letter, stage, combined)
- Derived labels CSVs
- NEW: Training curves (loss & accuracy) for Letter, Subclass (Combined), Stage, and Overall
- NEW (metrics): Top-1 / Top-2 accuracy and Macro Precision/Recall/F1 for Letter and Subclass (number)
"""

# =========================
# 0) Imports & Reproducibility
# =========================
import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import os, re, math, json, warnings
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
# --- NEW: macro metrics
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models, utils, callbacks, optimizers, regularizers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
tf.keras.backend.clear_session()

# =========================
# 1) Configuration
# =========================
ZIP_PATH = "star_images_bundle.zip"
IMG_SIZE = (256, 256)
MAX_IMAGES = None
TEST_SIZE = 0.15
VAL_SIZE = 0.15

BASE_BATCH_SIZE = 64                      ##### 128 for smaller dataset, 512 for larger one
AUTO_BATCH_SCALE = True
EPOCHS = 100

MIN_BRIGHT_FRAC = .1
CLUSTER_K = 6
SAVE_DERIVED_LABELS = True
CACHE_RAM_THRESHOLD_BYTES = 400 * 1024 * 1024  # ~400MB

# Image quality thresholds
QUALITY_MIN_PEAK = 12.0
QUALITY_MIN_PEAK_BG_RATIO = 1.5
QUALITY_MAX_CENTROID_OFFSET = 10.0
QUALITY_MAX_SAT_FRAC = 0.05
QUALITY_GRAYSCALE_MAD = 1.0
QUALITY_MIN_NONZERO_PIXELS = 3

MIN_METADATA_LABELS_TOTAL = 8
MIN_METADATA_DISTINCT_CLASSES = 2

KNOWN_SPECTRAL = ['O','B','A','F','G','K','M']
CANONICAL_LETTER_ORDER = ['O','B','A','F','G','K','M']

# Stage head classes
STAGE_CLASSES = ['white_dwarf','main_sequence','subgiant','giant','supergiant','unknown']
NUM_STAGE_CLASSES = len(STAGE_CLASSES)

# GPU / batch scaling
gpus = tf.config.list_physical_devices('GPU')
HAS_GPU = len(gpus) > 0
if AUTO_BATCH_SCALE and HAS_GPU:
    BATCH_SIZE = min(256, BASE_BATCH_SIZE * 2)
else:
    BATCH_SIZE = BASE_BATCH_SIZE

print(f"GPU present: {HAS_GPU} | Batch size: {BATCH_SIZE} | EPOCHS: {EPOCHS}")

# =========================
# 2) Color/CCT Helpers
# =========================
def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1.0 + a)) ** 2.4)

SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float32)

def rgb_to_xyz(rgb):
    arr = np.asarray(rgb, dtype=np.float32).reshape(3,)
    X, Y, Z = SRGB_TO_XYZ.dot(arr)
    return float(X), float(Y), float(Z)

def xyz_to_chromaticity(X, Y, Z):
    denom = X + Y + Z
    if denom <= 0: return None, None
    return X/denom, Y/denom

def chromaticity_to_cct(x, y):
    if x is None or y is None: return None
    denom = (y - 0.1858)
    if abs(denom) < 1e-12: return None
    n = (x - 0.3320) / denom
    cct = -449.0*(n**3) + 3525.0*(n**2) - 6823.3*n + 5520.33
    if np.isnan(cct) or cct <= 0: return None
    return float(cct)

def estimate_cct_from_image_array(img_arr):
    if img_arr is None: return None
    arr = img_arr.astype(np.float32)/255.0
    luma = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
    thr = max(np.percentile(luma, 90) if luma.size > 0 else 0.02, 0.02)
    mask = luma >= thr
    min_samples = max(5, int(arr.shape[0]*arr.shape[1]*MIN_BRIGHT_FRAC))
    if mask.sum() < min_samples:
        flat_idx = np.argsort(luma.flatten())[::-1]
        topk = min(len(flat_idx), min_samples)
        if topk <= 0: return None
        idx = flat_idx[:topk]
        y_idx = idx // arr.shape[1]; x_idx = idx % arr.shape[1]
        mask = np.zeros_like(luma, dtype=bool); mask[y_idx, x_idx] = True
    sampled = arr[mask]
    if sampled.size == 0: return None
    med_rgb = np.median(sampled, axis=0)
    lin_rgb = srgb_to_linear(med_rgb)
    try:
        X, Y, Z = rgb_to_xyz(lin_rgb)
    except Exception:
        return None
    if (X+Y+Z) <= 0: return None
    x, y = xyz_to_chromaticity(X, Y, Z)
    return chromaticity_to_cct(x, y)

def teff_to_spectral(teff):
    try:
        t = float(teff)
    except Exception:
        return None
    if t > 30000: return 'O'
    if t > 10000: return 'B'
    if t > 7500: return 'A'
    if t > 6000: return 'F'
    if t > 5200: return 'G'
    if t > 3700: return 'K'
    return 'M'

# =========================
# 3) I/O: Read ZIP & Metadata
# =========================
def read_metadata_from_zip(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            candidates = [n for n in zf.namelist() if os.path.basename(n).lower() in ('metadata.csv','labels.csv','metadata.txt','labels.txt')]
            if not candidates:
                candidates = [n for n in zf.namelist() if n.lower().endswith('.csv')]
            for name in candidates:
                try:
                    df = pd.read_csv(BytesIO(zf.read(name)))
                    return df
                except Exception:
                    continue
    except Exception:
        pass
    return None

def load_images_and_basename_list(zip_path, img_size=(256,256), max_images=None):
    imgs, basenames, paths = [], [], []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
        if max_images: names = names[:max_images]
        for n in names:
            try:
                with zf.open(n) as f:
                    img = Image.open(BytesIO(f.read())).convert('RGB')
                    resamp = Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else getattr(Image,'LANCZOS', Image.BICUBIC)
                    img = img.resize(img_size, resample=resamp)
                    arr = np.array(img)
                    imgs.append(arr)
                    basenames.append(os.path.basename(n))
                    paths.append(n)
            except Exception:
                continue
    if len(imgs) == 0:
        return np.zeros((0,img_size[0],img_size[1],3), dtype=np.uint8), [], []
    return np.array(imgs), basenames, paths

# =========================
# 4) Image Quality Metrics & Filtering
# =========================
def analyze_image_quality(img_arr, annulus_r_in=8, annulus_r_out=28, central_box=5):
    try:
        arr = img_arr.astype(float)
    except Exception:
        return None
    h,w = arr.shape[:2]
    cy, cx = (h-1)/2.0, (w-1)/2.0
    yy, xx = np.indices((h,w))
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    half = central_box//2
    y0 = int(round(cy))-half; y1 = y0 + 2*half + 1
    x0 = int(round(cx))-half; x1 = x0 + 2*half + 1
    y0, y1 = max(0,y0), min(h,y1); x0, x1 = max(0,x0), min(w,x1)
    central_patch = arr[y0:y1, x0:x1, :]
    central_peak = float(central_patch.max()) if central_patch.size>0 else float(arr.max())
    mask_ann = (r>=annulus_r_in) & (r<=annulus_r_out)
    if mask_ann.sum() == 0:
        ann_median = float(np.median(arr.mean(axis=2))) if arr.size>0 else 0.0
    else:
        ann_median = float(np.median(arr.mean(axis=2)[mask_ann]))
    peak_to_bg = central_peak / (ann_median + 1e-6)
    ch0, ch1, ch2 = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mad = (np.abs(ch0-ch1) + np.abs(ch0-ch2) + np.abs(ch1-ch2))/3.0
    mean_mad = float(np.nanmean(mad)) if mad.size>0 else 0.0
    sat_frac = float(((arr >= 250).all(axis=2)).sum())/(h*w) if (h*w)>0 else 0.0
    intensity = arr.mean(axis=2)
    thr = max(ann_median + 5.0, np.percentile(intensity,90)*0.5 if intensity.size>0 else 0.0)
    mask_bright = intensity >= thr
    bright_count = int(mask_bright.sum())
    if bright_count < 1:
        flat_idx = np.argsort(intensity.flatten())[::-1]
        topk = min(len(flat_idx), max(1, int(h*w*0.005)))
        if topk <= 0:
            centroid_y, centroid_x = cy, cx
        else:
            idx = flat_idx[:topk]; ys = idx // w; xs = idx % w
            centroid_y = float(ys.mean()) if len(ys)>0 else cy
            centroid_x = float(xs.mean()) if len(xs)>0 else cx
    else:
        coords = np.argwhere(mask_bright); ys = coords[:,0]; xs = coords[:,1]
        centroid_y, centroid_x = float(ys.mean()), float(xs.mean())
    centroid_offset = float(np.sqrt((centroid_x-cx)**2 + (centroid_y-cy)**2))
    nonzero_bright = int((intensity > (ann_median + 1.0)).sum())
    return {"central_peak":central_peak,"annulus_median":ann_median,"peak_to_bg":peak_to_bg,"mean_mad":mean_mad,"sat_frac":sat_frac,"centroid_offset":centroid_offset,"bright_count":bright_count,"nonzero_bright":nonzero_bright,"max_intensity":float(intensity.max()) if intensity.size>0 else 0.0}

def is_bad_image_by_metrics(metrics,
                            QUALITY_MIN_PEAK=QUALITY_MIN_PEAK,
                            QUALITY_MIN_PEAK_BG_RATIO=QUALITY_MIN_PEAK_BG_RATIO,
                            QUALITY_MAX_CENTROID_OFFSET=QUALITY_MAX_CENTROID_OFFSET,
                            QUALITY_MAX_SAT_FRAC=QUALITY_MAX_SAT_FRAC,
                            QUALITY_GRAYSCALE_MAD=QUALITY_GRAYSCALE_MAD,
                            QUALITY_MIN_NONZERO_PIXELS=QUALITY_MIN_NONZERO_PIXELS):
    reasons = []
    if metrics is None: return True, ["invalid_image"]
    if metrics["central_peak"] < QUALITY_MIN_PEAK:
        reasons.append(f"low_peak<{QUALITY_MIN_PEAK}")
    if metrics["peak_to_bg"] < QUALITY_MIN_PEAK_BG_RATIO:
        reasons.append(f"low_peak_bg_ratio<{QUALITY_MIN_PEAK_BG_RATIO:.2f}")
    if metrics["centroid_offset"] > QUALITY_MAX_CENTROID_OFFSET:
        reasons.append(f"offcenter>{QUALITY_MAX_CENTROID_OFFSET}px")
    if metrics["sat_frac"] > QUALITY_MAX_SAT_FRAC:
        reasons.append(f"saturated_frac>{QUALITY_MAX_SAT_FRAC:.3f}")
    if metrics["mean_mad"] < QUALITY_GRAYSCALE_MAD:
        reasons.append(f"effectively_grayscale(mad<{QUALITY_GRAYSCALE_MAD})")
    if metrics["nonzero_bright"] < QUALITY_MIN_NONZERO_PIXELS:
        reasons.append(f"too_few_bright_pixels<{QUALITY_MIN_NONZERO_PIXELS}")
    is_bad = len(reasons) > 0
    return is_bad, reasons

def filter_images_for_training(imgs, basenames,
                               QUALITY_MIN_PEAK=QUALITY_MIN_PEAK,
                               QUALITY_MIN_PEAK_BG_RATIO=QUALITY_MIN_PEAK_BG_RATIO,
                               QUALITY_MAX_CENTROID_OFFSET=QUALITY_MAX_CENTROID_OFFSET,
                               QUALITY_MAX_SAT_FRAC=QUALITY_MAX_SAT_FRAC,
                               QUALITY_GRAYSCALE_MAD=QUALITY_GRAYSCALE_MAD,
                               QUALITY_MIN_NONZERO_PIXELS=QUALITY_MIN_NONZERO_PIXELS):
    ok_imgs, ok_names, bad_rows, metrics_list = [], [], [], []
    for i,(img,name) in enumerate(zip(imgs, basenames)):
        metrics = analyze_image_quality(img)
        metrics_list.append(metrics)
        bad,reasons = is_bad_image_by_metrics(metrics, QUALITY_MIN_PEAK, QUALITY_MIN_PEAK_BG_RATIO,
                                             QUALITY_MAX_CENTROID_OFFSET, QUALITY_MAX_SAT_FRAC,
                                             QUALITY_GRAYSCALE_MAD, QUALITY_MIN_NONZERO_PIXELS)
        if bad:
            row = {"filename": name, "index": i, "reasons": ";".join(reasons)}
            if metrics:
                for k,v in metrics.items(): row[k]=v
            bad_rows.append(row)
        else:
            ok_imgs.append(img); ok_names.append(name)
    bad_df = pd.DataFrame(bad_rows)
    return np.array(ok_imgs), ok_names, bad_df, metrics_list

def adaptive_filter_wrapper(imgs, basenames, orig_count):
    p = {"QUALITY_MIN_PEAK":QUALITY_MIN_PEAK,"QUALITY_MIN_PEAK_BG_RATIO":QUALITY_MIN_PEAK_BG_RATIO,"QUALITY_MAX_CENTROID_OFFSET":QUALITY_MAX_CENTROID_OFFSET,"QUALITY_MAX_SAT_FRAC":QUALITY_MAX_SAT_FRAC,"QUALITY_GRAYSCALE_MAD":QUALITY_GRAYSCALE_MAD,"QUALITY_MIN_NONZERO_PIXELS":QUALITY_MIN_NONZERO_PIXELS}
    target_keep = max(16, int(0.35 * orig_count))
    last_good=None; last_bad_df=None
    for it in range(5):
        print(f"Quality filter attempt {it+1} with thresholds: min_peak={p['QUALITY_MIN_PEAK']:.2f}, peak_bg={p['QUALITY_MIN_PEAK_BG_RATIO']:.2f}, centroid_off={p['QUALITY_MAX_CENTROID_OFFSET']:.2f}, sat_frac={p['QUALITY_MAX_SAT_FRAC']:.3f}, mad={p['QUALITY_GRAYSCALE_MAD']:.2f}, min_bright={p['QUALITY_MIN_NONZERO_PIXELS']}")
        ok_imgs, ok_names, bad_df, _ = filter_images_for_training(imgs, basenames,
                                                                 QUALITY_MIN_PEAK=p['QUALITY_MIN_PEAK'],
                                                                 QUALITY_MIN_PEAK_BG_RATIO=p['QUALITY_MIN_PEAK_BG_RATIO'],
                                                                 QUALITY_MAX_CENTROID_OFFSET=p['QUALITY_MAX_CENTROID_OFFSET'],
                                                                 QUALITY_MAX_SAT_FRAC=p['QUALITY_MAX_SAT_FRAC'],
                                                                 QUALITY_GRAYSCALE_MAD=p['QUALITY_GRAYSCALE_MAD'],
                                                                 QUALITY_MIN_NONZERO_PIXELS=p['QUALITY_MIN_NONZERO_PIXELS'])
        kept = len(ok_imgs)
        print(f" -> kept {kept}/{orig_count} images")
        if kept >= target_keep or it == 4:
            last_good=(ok_imgs, ok_names); last_bad_df=bad_df; break
        p['QUALITY_MIN_PEAK'] *= 0.7
        p['QUALITY_MIN_PEAK_BG_RATIO'] *= 0.7
        p['QUALITY_MAX_CENTROID_OFFSET'] *= 1.3
        p['QUALITY_MAX_SAT_FRAC'] *= 1.5
        p['QUALITY_GRAYSCALE_MAD'] *= 0.6
        p['QUALITY_MIN_NONZERO_PIXELS'] = max(1, int(p['QUALITY_MIN_NONZERO_PIXELS']*0.7))
        last_good=(ok_imgs, ok_names); last_bad_df=bad_df
    return last_good[0], last_good[1], last_bad_df

# =========================
# 5) Metadata Filename Matching
# =========================
def build_image_lookup(basenames):
    return {b.lower(): b for b in basenames}

def try_match_meta_fname_to_image(fn_value, image_lookup):
    if fn_value is None: return None
    s = str(fn_value).strip()
    if not s: return None
    candidates=[]
    b = os.path.basename(s).strip()
    if b: candidates.append(b)
    candidates.append(s)
    candidates.append(s.lower())
    parts = re.split(r'[\\/]', s)
    for p in parts:
        if p: candidates.append(p)
    candidates.append(re.sub(r'^dataset/','', s, flags=re.I))
    candidates.append(re.sub(r'^dataset/images/','', s, flags=re.I))
    m = re.search(r'(\d+)', s)
    if m:
        digits = m.group(1)
        candidates.append(digits)
        candidates.append(f"star_{int(digits):05d}.png")
        candidates.append(f"star_{digits}.png")
    for c in candidates:
        k = c.lower()
        if k in image_lookup: return image_lookup[k]
    return None

# =========================
# 6) Plotting Helper (Confusion Matrices)
# =========================
def plot_and_save_confusion(cm, labels_list, title, outname):
    plt.figure(figsize=(max(6, len(labels_list)*0.6), max(5, len(labels_list)*0.6)))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels_list)), labels_list, rotation=45)
    plt.yticks(range(len(labels_list)), labels_list)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    try:
        plt.savefig(outname, dpi=200)
        print(f"Saved confusion matrix -> {outname}")
    except Exception as e:
        print("Could not save confusion matrix:", e)
    plt.show()

# =========================
# 7) Load Images + Adaptive Filtering
# =========================
print("Loading images from ZIP...")
X_imgs, basenames, paths = load_images_and_basename_list(ZIP_PATH, IMG_SIZE, MAX_IMAGES)
orig_n_images = len(X_imgs)
print(f"Loaded {orig_n_images} images")

print("Adaptive quality filtering ...")
X_ok, basenames_ok, bad_df = adaptive_filter_wrapper(X_imgs, basenames, orig_n_images)
kept = len(X_ok)
print(f"Final kept {kept}/{orig_n_images} images")

if len(bad_df) > 0:
    try: bad_df.to_csv("bad_images.csv", index=False); print("Wrote bad_images.csv")
    except Exception as e: print("Could not write bad_images.csv:", e)

X_imgs = X_ok; basenames = basenames_ok; n_images = len(X_imgs)
if n_images == 0:
    raise RuntimeError("No images left after filtering.")

# =========================
# 8) Read Metadata (if present) and Estimate CCT
# =========================
print("Reading metadata (if present)...")
meta_df = read_metadata_from_zip(ZIP_PATH)
if meta_df is not None:
    try: print("Found metadata preview:\n", meta_df.head())
    except: pass
else:
    print("No metadata found.")

cct_list = [None]*n_images
for i,img in enumerate(X_imgs):
    try: cct_list[i] = estimate_cct_from_image_array(img)
    except: cct_list[i] = None

spec_from_cct = [teff_to_spectral(c) if c is not None else None for c in cct_list]
print("Sample estimated CCTs (first 10):")
for b,c,s in zip(basenames[:10], cct_list[:10], spec_from_cct[:10]):
    print(" ", b, "CCT=", f"{c:.1f}" if c is not None else "None", "->", s)

# =========================
# 9) Label Derivation (metadata preferred, else color-derived)
# =========================
use_metadata_for_training = False
labels = [None]*n_images
subclasses_from_meta = [None]*n_images
subclass_confidence_flag = [0]*n_images

if meta_df is not None:
    image_lookup = build_image_lookup(basenames)
    meta_cols_lower = {c.lower():c for c in meta_df.columns}
    spec_col_name = None
    for candidate in ('spectral_primary_letter','spectral_type_letter','spectral_type','spectral','spectralclass','spectral_class','spectraltype','spec','type','label'):
        if candidate in meta_cols_lower:
            spec_col_name = meta_cols_lower[candidate]; break
    fname_candidates = [c for c in meta_df.columns if c.lower() in ('filename','file','raw_file','image','image_name','proc_filename','path')]
    subclass_candidates = [c for c in meta_df.columns if c.lower() in ('spectral_subclass','subclass','spec_subclass','sptype_subclass')]
    if spec_col_name and fname_candidates:
        fname_col = fname_candidates[0]
        mapping_letter = {}
        mapping_sub = {}
        for _, row in meta_df.iterrows():
            fn = row.get(fname_col,"")
            matched = try_match_meta_fname_to_image(fn, image_lookup)
            if not matched:
                for c in meta_df.columns:
                    if c == fname_col: continue
                    val = row.get(c,"")
                    candidate_match = try_match_meta_fname_to_image(val, image_lookup)
                    if candidate_match:
                        matched = candidate_match; break
            if not matched: continue
            lbl = row.get(spec_col_name, "")
            if pd.isna(lbl) or str(lbl).strip() == "": continue
            m = re.search(r'([OBAFGKM])', str(lbl).upper())
            if m: mapping_letter[matched] = m.group(1)
            ssub = None
            for sc in subclass_candidates:
                try:
                    sv = row.get(sc, None)
                    if sv is not None and not pd.isna(sv) and str(sv).strip() != "":
                        ssub = int(round(float(sv))); break
                except Exception:
                    pass
            if ssub is None:
                def parse_subclass_from_sptype_str(s):
                    if not s or not isinstance(s, str): return None
                    mm = re.search(r'([OBAFGKM])\s*([0-9](?:\.[0-9])?)', s.upper())
                    if mm:
                        try: return int(round(float(mm.group(2))))
                        except: return None
                    mm2 = re.search(r'^[OBAFGKM]\s*([0-9](?:\.[0-9])?)', s.upper())
                    if mm2:
                        try: return int(round(float(mm2.group(1))))
                        except: return None
                    return None
                psub = parse_subclass_from_sptype_str(str(lbl))
                if psub is not None: ssub = psub
            if ssub is not None:
                mapping_sub[matched] = max(0,min(9,int(ssub)))
        mapped_labels = [mapping_letter.get(b, None) for b in basenames]
        mapped_count = sum(1 for l in mapped_labels if l is not None)
        mapped_distinct = set(l for l in mapped_labels if l is not None)
        print(f"Metadata mapping found {mapped_count}/{n_images} labels across {len(mapped_distinct)} classes: {mapped_distinct}")
        if mapped_count >= MIN_METADATA_LABELS_TOTAL and len(mapped_distinct) >= MIN_METADATA_DISTINCT_CLASSES:
            labels = mapped_labels
            for i,b in enumerate(basenames):
                if b in mapping_sub:
                    subclasses_from_meta[i] = mapping_sub[b]; subclass_confidence_flag[i] = 1
            use_metadata_for_training = True
            print("Using metadata labels for training.")
        else:
            print("Metadata insufficient -> fallback.")
    else:
        print("Metadata present but couldn't find filename & spectral columns reliably; fallback.")

if not use_metadata_for_training:
    print("Falling back to color-derived labeling.")
    for i, s in enumerate(spec_from_cct):
        if s in KNOWN_SPECTRAL: labels[i] = s

    def rgb_feature_from_img(img):
        arr = img.astype(np.float32)/255.0
        luma = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
        thr = max(np.percentile(luma,90) if luma.size>0 else 0.02, 0.02)
        mask = luma >= thr
        if mask.sum() < 3:
            idx = np.argmax(luma); y = idx // arr.shape[1]; x = idx % arr.shape[1]; return arr[y,x]
        sampled = arr[mask]
        if sampled.size == 0: return arr[arr.shape[0]//2, arr.shape[1]//2]
        return sampled.mean(axis=0)

    missing_idxs = [i for i,l in enumerate(labels) if l is None]
    if missing_idxs:
        all_feats = np.array([rgb_feature_from_img(img) for img in X_imgs])
        k = min(CLUSTER_K, max(2, n_images//50))
        try: km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(all_feats)
        except Exception: km = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit(all_feats)
        centroid_labels = []
        for c_idx in range(k):
            centroid_rgb = km.cluster_centers_[c_idx]
            centroid_lin = srgb_to_linear(centroid_rgb)
            try:
                Xc,Yc,Zc = rgb_to_xyz(centroid_lin)
                if (Xc+Yc+Zc)>0:
                    xc, yc = Xc/(Xc+Yc+Zc), Yc/(Xc+Yc+Zc)
                else: xc, yc = None, None
            except Exception: xc, yc = None, None
            cct_cent = chromaticity_to_cct(xc,yc)
            lbl_cent = teff_to_spectral(cct_cent) if cct_cent is not None else None
            centroid_labels.append(lbl_cent)
        for i in missing_idxs:
            cluster = int(km.labels_[i]); labels[i] = centroid_labels[cluster]
    dist = Counter(labels)
    if len(dist) == 1 or (max(dist.values())/max(1,len(labels)) > 0.75):
        print("Detected collapse to dominant class; forcing diversification via clustering.")
        all_feats = np.array([rgb_feature_from_img(img) for img in X_imgs])
        desired_k = min(max(3, len(KNOWN_SPECTRAL)), max(2, n_images//100, CLUSTER_K))
        desired_k = max(2, desired_k)
        try: km3 = KMeans(n_clusters=desired_k, random_state=RANDOM_STATE, n_init=10).fit(all_feats)
        except Exception: km3 = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10).fit(all_feats)
        centroids = km3.cluster_centers_
        br = [ (c[2]+1e-6)/(c[0]+1e-6) for c in centroids ]
        order = np.argsort(br)
        mapped = {}
        for rank_index, cent_idx in enumerate(order):
            frac = rank_index / max(1,(len(order)-1))
            spec_idx = int(round(frac*(len(KNOWN_SPECTRAL)-1)))
            mapped[cent_idx] = KNOWN_SPECTRAL[max(0,min(len(KNOWN_SPECTRAL)-1,spec_idx))]
        for i in range(n_images):
            cluster = int(km3.labels_[i]); labels[i] = mapped.get(cluster, 'A')
        print("New label distribution:", Counter(labels))
    for i,l in enumerate(labels):
        if l not in KNOWN_SPECTRAL: labels[i] = 'A'
    print("Color-derived label distribution:", Counter(labels))

# =========================
# 10) Build Labeled Arrays + Subclass Estimation
# =========================
labeled_indices = [i for i,l in enumerate(labels) if l in KNOWN_SPECTRAL]
print(f"Images with usable labels: {len(labeled_indices)}/{n_images}")
if len(labeled_indices) == 0:
    raise RuntimeError("No usable labels.")

X_all = np.array([X_imgs[i] for i in labeled_indices])
filenames_all = [basenames[i] for i in labeled_indices]
labels_all = [labels[i] for i in labeled_indices]
cct_all = [cct_list[i] for i in labeled_indices]
subclasses_meta_all = [subclasses_from_meta[i] for i in labeled_indices]
subclass_conf_meta_all = [subclass_confidence_flag[i] for i in labeled_indices]

CLASS_TEMP_RANGES = {
    "O": (50000.0, 30000.0),
    "B": (30000.0, 10000.0),
    "A": (10000.0, 7500.0),
    "F": (7500.0, 6000.0),
    "G": (6000.0, 5200.0),
    "K": (5200.0, 3700.0),
    "M": (3700.0, 2400.0)
}
def temp_to_letter_and_subclass(temp_k):
    if temp_k is None: return None, None
    T = float(temp_k)
    for letter, (thigh, tlow) in CLASS_TEMP_RANGES.items():
        if T <= thigh + 1 and T >= tlow - 1:
            denom = (thigh - tlow) if (thigh - tlow) != 0 else 1.0
            frac = (thigh - T) / denom
            sub = int(round(max(0, min(9, frac*9.0))))
            return letter, sub
    if T > max(v[0] for v in CLASS_TEMP_RANGES.values()): return "O", 0
    if T < min(v[1] for v in CLASS_TEMP_RANGES.values()): return "M", 9
    return None, None

subclass_all = []
subclass_confidence = []
for lab_letter,cct,meta_sub,meta_conf in zip(labels_all, cct_all, subclasses_meta_all, subclass_conf_meta_all):
    if meta_sub is not None and meta_conf == 1:
        subclass_all.append(int(meta_sub)); subclass_confidence.append(1); continue
    if cct is not None:
        lett, sub = temp_to_letter_and_subclass(cct)
        if lett is not None and sub is not None:
            if lett == lab_letter:
                subclass_all.append(int(sub)); subclass_confidence.append(1); continue
            else:
                thigh, tlow = CLASS_TEMP_RANGES.get(lab_letter, (None,None))
                if thigh is not None and tlow is not None:
                    denom = (thigh - tlow) if (thigh - tlow) != 0 else 1.0
                    frac = (thigh - float(cct))/denom
                    sub_est = int(round(max(0,min(9,frac*9.0))))
                    subclass_all.append(sub_est); subclass_confidence.append(1); continue
    subclass_all.append(5); subclass_confidence.append(0)

subclass_all = [int(max(0,min(9,s))) for s in subclass_all]
print("Label counts before balancing:", Counter(labels_all))
print("Subclass counts (raw):", Counter(subclass_all))

# =========================
# 11) Curved Class Balancing
# =========================
rng = np.random.RandomState(RANDOM_STATE)
counts = Counter(labels_all)
classes = sorted(list(counts.keys()))
total_images = len(labels_all)
if len(classes) == 0:
    raise RuntimeError("No classes present.")

BALANCE_BETA = 0.6
cls_counts = np.array([counts[cls] for cls in classes], dtype=float)
transformed = np.power(cls_counts, BALANCE_BETA)
scale = total_images / transformed.sum() if transformed.sum()>0 else 1.0
raw_targets = transformed * scale
floored = np.floor(raw_targets).astype(int)
remainder = int(total_images - floored.sum())
frac = raw_targets - floored
order_frac = np.argsort(-frac)
targets = floored.copy()
idx_order = 0
while remainder > 0:
    targets[order_frac[idx_order % len(order_frac)]] += 1
    idx_order += 1; remainder -= 1
for i in range(len(targets)):
    if targets[i] < 1: targets[i] = 1
diff = int(total_images - targets.sum())
if diff != 0:
    order_by_orig = np.argsort(-cls_counts); k = 0
    while diff != 0:
        idx = order_by_orig[k % len(order_by_orig)]
        if diff > 0:
            targets[idx] += 1; diff -= 1
        else:
            if targets[idx] > 1:
                targets[idx] -= 1; diff += 1
        k += 1
target_map = {cls: int(targets[i]) for i, cls in enumerate(classes)}
print("Original counts:", dict(zip(classes, cls_counts.astype(int))))
print("Curved target counts:", target_map)

selected_idx = []
for cls in classes:
    idxs = [i for i,lab in enumerate(labels_all) if lab==cls]
    tgt = target_map[cls]
    if len(idxs) == 0: continue
    if tgt <= len(idxs):
        chosen = rng.choice(idxs, size=tgt, replace=False).tolist()
    else:
        chosen = rng.choice(idxs, size=tgt, replace=True).tolist()
    selected_idx.extend(chosen)

rng.shuffle(selected_idx)
X_bal = np.array([X_all[i] for i in selected_idx])
filenames_bal = [filenames_all[i] for i in selected_idx]
labels_bal = [labels_all[i] for i in selected_idx]
cct_bal = [cct_all[i] for i in selected_idx]
subclass_bal = [subclass_all[i] for i in selected_idx]
subclass_conf_bal = [subclass_confidence[i] for i in selected_idx]

print("Label counts after balancing:", Counter(labels_bal))

# Ensure at least 2 per class for stratify
counts_bal = Counter(labels_bal)
need_dup = [cls for cls,cnt in counts_bal.items() if cnt < 2]
if need_dup:
    print("Duplicating small classes to ensure >=2 for stratify:", need_dup)
    X_list = list(X_bal); fname_list = list(filenames_bal); lab_list = list(labels_bal)
    cct_list_new = list(cct_bal); sub_list_new = list(subclass_bal); subconf_new = list(subclass_conf_bal)
    def small_augment(img):
        arr = img.astype(np.float32)
        scale = 1.0 + (0.03 * np.random.randn())
        arr2 = np.clip(arr * scale, 0, 255)
        dy = np.random.randint(-1,2); dx = np.random.randint(-1,2)
        arr2 = np.roll(np.roll(arr2, dy, axis=0), dx, axis=1)
        return arr2.astype(np.uint8)
    for cls in need_dup:
        idxs = [i for i,L in enumerate(lab_list) if L==cls]
        if not idxs: continue
        idx0 = idxs[0]
        X_list.append(small_augment(X_bal[idx0])); fname_list.append(filenames_bal[idx0])
        lab_list.append(lab_list[idx0]); cct_list_new.append(cct_bal[idx0]); sub_list_new.append(subclass_bal[idx0]); subconf_new.append(subclass_conf_bal[idx0])
    X_bal = np.array(X_list)
    filenames_bal = fname_list; labels_bal = lab_list
    cct_bal = cct_list_new; subclass_bal = sub_list_new; subconf_new = subconf_new
    print("Counts after duplication:", Counter(labels_bal))

X_final = X_bal; filenames_final = filenames_bal; labels_final = labels_bal
cct_final = cct_bal; subclass_final = subclass_bal; subclass_conf_final = subclass_conf_bal
print("Final label distribution:", Counter(labels_final))

if SAVE_DERIVED_LABELS:
    df_out = pd.DataFrame({
        "filename":filenames_final,
        "label_used_for_training":labels_final,
        "estimated_CCT":cct_final,
        "subclass_used_for_training":subclass_final,
        "subclass_confidence":subclass_conf_final
    })
    try: df_out.to_csv("derived_labels_used_for_training_balanced_curved_multitask.csv", index=False); print("Saved derived labels CSV.")
    except Exception as e: print("Could not save derived labels CSV:", e)

# =========================
# 12) Encode Labels (letters, combined letter*10+subclass, stage)
# =========================
le = LabelEncoder()
y_letters_enc = le.fit_transform(labels_final)
num_letter_classes = len(le.classes_)
y_sub_all = np.array(subclass_final, dtype=int)
subclass_confidence_arr = np.array(subclass_conf_final, dtype=int)

num_combined_classes = num_letter_classes * 10
y_combined_all = (y_letters_enc * 10 + y_sub_all).astype(int)

stage_labels_default = ['unknown'] * len(filenames_final)
stage_map_from_meta = {}
if meta_df is not None:
    image_lookup_final = build_image_lookup(filenames_final)
    meta_cols_lower = {c.lower():c for c in meta_df.columns}
    stage_col_candidates = [c for c in meta_df.columns if c.lower() in ('stellar_stage','stage','lumclass','luminosity_class','spectral_luminosity','luminosity','class')]
    def canonical_stage_from_str(s):
        if s is None: return None
        ss = str(s).strip().lower()
        if ss == "": return None
        if 'white' in ss or 'wd' in ss or 'dwarf' in ss: return 'white_dwarf'
        if 'main' in ss or 'v' == ss or ss.startswith('v'): return 'main_sequence'
        if 'sub' in ss or 'iv' in ss: return 'subgiant'
        if 'super' in ss or ss.startswith('i '): return 'supergiant'
        if 'iii' in ss or 'giant' in ss or 'iii' in ss: return 'giant'
        if 'iv' in ss or 'iv' == ss: return 'subgiant'
        if 'v' in ss and len(ss) <= 2: return 'main_sequence'
        return None
    if stage_col_candidates:
        sc = stage_col_candidates[0]
        for _, row in meta_df.iterrows():
            fn = row.get('filename', None) if 'filename' in meta_df.columns else None
            if fn is None:
                for c in meta_df.columns:
                    if c.lower() in ('file','raw_file','path','proc_filename','image','image_name'):
                        fn = row.get(c, None); break
            matched = try_match_meta_fname_to_image(fn, image_lookup_final)
            if not matched:
                for c in meta_df.columns:
                    if c.lower() in ('filename','file','raw_file','path','proc_filename','image','image_name'):
                        continue
                    candidate_val = row.get(c, None)
                    mm = try_match_meta_fname_to_image(candidate_val, image_lookup_final)
                    if mm:
                        matched = mm; break
            if not matched:
                continue
            st_raw = row.get(sc, None)
            canon = canonical_stage_from_str(st_raw)
            if canon:
                stage_map_from_meta[matched] = canon

for i,fn in enumerate(filenames_final):
    if fn in stage_map_from_meta:
        stage_labels_default[i] = stage_map_from_meta[fn]

non_unknown_count = sum(1 for s in stage_labels_default if s != 'unknown')
distinct_stage_labels = set(s for s in stage_labels_default if s != 'unknown')
print("Stage labels from metadata found:", non_unknown_count, "distinct:", distinct_stage_labels)

stage_training_enabled = (non_unknown_count > 0 and len(distinct_stage_labels) > 0)
print("Stage head training enabled:", stage_training_enabled)

stage_to_idx = {s:i for i,s in enumerate(STAGE_CLASSES)}
y_stage_all = np.array([stage_to_idx.get(s if s is not None else 'unknown', stage_to_idx['unknown']) for s in stage_labels_default], dtype=int)

# =========================
# 13) Sample Weights (letters, subclass, stage, combined)
# =========================
try:
    cw = compute_class_weight('balanced', classes=np.unique(y_letters_enc), y=y_letters_enc)
    class_weight_dict = dict(zip(np.unique(y_letters_enc), cw))
except Exception:
    class_weight_dict = {int(i):1.0 for i in np.unique(y_letters_enc)}
sample_weight_letter_all = np.array([float(class_weight_dict.get(int(lbl), 1.0)) for lbl in y_letters_enc], dtype=float)
sample_weight_letter_all = np.clip(sample_weight_letter_all, 0.01, 10.0)

try:
    cw_sub = compute_class_weight('balanced', classes=np.arange(10), y=y_sub_all)
    cw_sub = np.array(cw_sub, dtype=float)
    if cw_sub.shape[0] != 10:
        tmp = np.ones(10, dtype=float)
        for i, val in enumerate(cw_sub[:10]): tmp[i] = val
        cw_sub = tmp
except Exception:
    cw_sub = np.ones(10, dtype=float)
sample_weight_sub_all = np.array([ (cw_sub[s]) if conf==1 else 0.5 for s,conf in zip(y_sub_all, subclass_confidence_arr)], dtype=float)
sample_weight_sub_all = np.clip(sample_weight_sub_all, 0.01, 50.0)

try:
    unique_stages = np.unique(y_stage_all)
    if len(unique_stages) > 1:
        cw_stage_vals = compute_class_weight('balanced', classes=unique_stages, y=y_stage_all)
        cw_stage_dict = {int(c): float(w) for c,w in zip(unique_stages, cw_stage_vals)}
    else:
        cw_stage_dict = {int(c): 1.0 for c in unique_stages}
except Exception:
    cw_stage_dict = {int(c): 1.0 for c in np.unique(y_stage_all)}
sample_weight_stage_all = np.array([cw_stage_dict.get(int(lbl), 1.0) for lbl in y_stage_all], dtype=float)
sample_weight_stage_all = np.clip(sample_weight_stage_all, 0.01, 50.0)

sw_combined = np.power(sample_weight_letter_all * (sample_weight_sub_all + 1e-8) * (sample_weight_stage_all + 1e-8), 1/3)
sw_combined = sw_combined / np.mean(sw_combined)

# =========================
# 14) Train/Val/Test Splits + Normalization
# =========================
n_total = len(y_letters_enc)
if n_total < 3:
    raise RuntimeError("Not enough data to split.")

X_train, X_test, y_train, y_test, ysub_train, ysub_test, ycomb_train, ycomb_test, ystage_train, ystage_test, sw_train, sw_test = train_test_split(
    X_final, y_letters_enc, y_sub_all, y_combined_all, y_stage_all, sw_combined,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_letters_enc)

val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
if val_frac <= 0:
    X_val = np.zeros((0,IMG_SIZE[0],IMG_SIZE[1],3), dtype=X_train.dtype)
    y_val = np.array([]); ysub_val = np.array([]); ycomb_val = np.array([]); ystage_val = np.array([]); sw_val = np.array([])
else:
    X_train, X_val, y_train, y_val, ysub_train, ysub_val, ycomb_train, ycomb_val, ystage_train, ystage_val, sw_train, sw_val = train_test_split(
        X_train, y_train, ysub_train, ycomb_train, ystage_train, sw_train, test_size=val_frac, random_state=RANDOM_STATE, stratify=y_train)

print("Final splits -> train:", len(X_train), "val:", len(X_val), "test:", len(X_test))

X_train_norm = X_train.astype('float32')/255.0
X_val_norm   = X_val.astype('float32')/255.0
X_test_norm  = X_test.astype('float32')/255.0

# =========================
# 15) Baseline Logistic (RGB summary)
# =========================
if len(np.unique(y_train))>1 and len(X_train)>0:
    def extract_rgb_features(imgs):
        means = imgs.mean(axis=(1,2))
        stds  = imgs.std(axis=(1,2))
        return np.hstack([means,stds])
    try:
        X_train_feats = extract_rgb_features(X_train)
        X_val_feats = extract_rgb_features(X_val) if len(X_val)>0 else X_train_feats[:len(X_val)]
        X_test_feats = extract_rgb_features(X_test) if len(X_test)>0 else X_train_feats[:len(X_test)]
        clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='saga', C=1.0)
        clf.fit(X_train_feats, y_train)
        val_acc = clf.score(X_val_feats, y_val) if len(X_val_feats)>0 else np.nan
        test_acc = clf.score(X_test_feats, y_test) if len(X_test_feats)>0 else np.nan
        print("\nBaseline logistic accuracy (RGB summary):", float(test_acc))
        preds_baseline = clf.predict(X_test_feats) if len(X_test_feats)>0 else np.array([])
        if len(preds_baseline)>0:
            present_letters = list(le.classes_)
            ordered_present = [c for c in CANONICAL_LETTER_ORDER if c in present_letters]
            if len(ordered_present) == 0:
                order_indices = sorted(list(np.unique(y_test)))
            else:
                order_indices = [int(np.where(le.classes_ == c)[0][0]) for c in ordered_present]
            cm_base = confusion_matrix(y_test, preds_baseline, labels=order_indices)
            plot_and_save_confusion(cm_base, [le.classes_[i] for i in order_indices], 'Baseline Confusion Matrix (Letters)', 'baseline_confusion_matrix_letters.png')
        print("Baseline acc (val/test):", val_acc, test_acc)
    except Exception as e:
        print("Baseline logistic failed:", e)

# =========================
# 16) tf.data Datasets (with scalar sample weights)
# =========================
def make_dataset_with_scalar_sample_weight(X, y_letter_cat, y_combined_cat, y_stage_cat, sw_scalar, batch_size=BATCH_SIZE, training=False):
    X = X.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((X, {'letter_output': y_letter_cat, 'combined_output': y_combined_cat, 'stage_output': y_stage_cat}, sw_scalar.astype('float32')))
    dataset_bytes = X.nbytes if hasattr(X,'nbytes') else X.size*X.itemsize
    if training and dataset_bytes <= CACHE_RAM_THRESHOLD_BYTES:
        ds = ds.cache()
        print(f"Caching train dataset in RAM (~{dataset_bytes/1024/1024:.1f} MB)")
    if training:
        ds = ds.shuffle(buffer_size=max(1024, len(X)), seed=RANDOM_STATE, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

y_train_letter_cat = utils.to_categorical(y_train, num_letter_classes)
y_val_letter_cat = utils.to_categorical(y_val, num_letter_classes) if len(y_val)>0 else np.zeros((len(y_val), num_letter_classes), dtype=np.float32)
y_test_letter_cat = utils.to_categorical(y_test, num_letter_classes) if len(y_test)>0 else np.zeros((len(y_test), num_letter_classes), dtype=np.float32)

y_train_combined_cat = utils.to_categorical(ycomb_train, num_combined_classes)
y_val_combined_cat = utils.to_categorical(ycomb_val, num_combined_classes) if len(ycomb_val)>0 else np.zeros((len(y_val), num_combined_classes), dtype=np.float32)
y_test_combined_cat = utils.to_categorical(ycomb_test, num_combined_classes) if len(y_test)>0 else np.zeros((len(y_test), num_combined_classes), dtype=np.float32)

y_train_stage_cat = utils.to_categorical(ystage_train, NUM_STAGE_CLASSES)
y_val_stage_cat = utils.to_categorical(ystage_val, NUM_STAGE_CLASSES) if len(ystage_val)>0 else np.zeros((len(y_val), NUM_STAGE_CLASSES), dtype=np.float32)
y_test_stage_cat = utils.to_categorical(ystage_test, NUM_STAGE_CLASSES) if len(y_test)>0 else np.zeros((len(y_test), NUM_STAGE_CLASSES), dtype=np.float32)

train_ds = make_dataset_with_scalar_sample_weight(X_train_norm, y_train_letter_cat, y_train_combined_cat, y_train_stage_cat, sw_train, batch_size=BATCH_SIZE, training=True)
val_ds = None
if len(X_val_norm) > 0:
    val_ds = make_dataset_with_scalar_sample_weight(X_val_norm, y_val_letter_cat, y_val_combined_cat, y_val_stage_cat, sw_val, batch_size=BATCH_SIZE, training=False)

# =========================
# 17) CNN Model (multitask with letter, combined, stage outputs)
# =========================
def conv_block(x, filters, kernel=3, pool=True):
    x = layers.Conv2D(filters, (kernel,kernel), padding='same', kernel_regularizer=regularizers.l2(1e-5), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if pool:
        x = layers.MaxPooling2D((2,2))(x)
    return x

def build_simpler_multitask_combined(input_shape, num_letters, num_combined, num_stages):
    inp = layers.Input(shape=input_shape, name='input')
    x = inp
    x = conv_block(x, 32, kernel=3, pool=True)
    x = conv_block(x, 64, kernel=3, pool=True)
    x = conv_block(x, 128, kernel=3, pool=True)
    x = layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-5), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    letter = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    letter = layers.Dropout(0.2)(letter)
    letter_out = layers.Dense(num_letters, activation='softmax', name='letter_output', dtype='float32')(letter)

    comb = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    comb = layers.Dropout(0.25)(comb)
    combined_out = layers.Dense(num_combined, activation='softmax', name='combined_output', dtype='float32')(comb)

    st = layers.Dense(96, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    st = layers.Dropout(0.25)(st)
    stage_out = layers.Dense(num_stages, activation='softmax', name='stage_output', dtype='float32')(st)

    model = models.Model(inputs=inp, outputs=[letter_out, combined_out, stage_out])
    return model

cnn = build_simpler_multitask_combined((IMG_SIZE[0], IMG_SIZE[1], 3), num_letter_classes, num_combined_classes, NUM_STAGE_CLASSES)
cnn.summary()

# =========================
# 18) Staged Training (heads then multitask)
# =========================
stage1_epochs = max(3, min(100, EPOCHS // 3))
stage2_epochs = max(1, EPOCHS - stage1_epochs)
print(f"Stage1 (head-only) epochs: {stage1_epochs}, Stage2 (fine-tune) epochs: {stage2_epochs}")

def infer_mode_for_monitor(monitor_name):
    if monitor_name is None: return 'min'
    mn = monitor_name.lower()
    if 'acc' in mn or 'accuracy' in mn: return 'max'
    return 'min'

def compile_for_stage(model, lr=1e-2, loss_weights={'letter_output':1.0, 'combined_output':0.0, 'stage_output':0.0}):
    losses  = {'letter_output': 'categorical_crossentropy', 'combined_output': 'categorical_crossentropy', 'stage_output': 'categorical_crossentropy'}
    metrics = {'letter_output': 'accuracy',                'combined_output': 'accuracy',                'stage_output': 'accuracy'}
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics)
    return model

monitor_metric_stage1 = 'val_letter_output_loss' if val_ds is not None else 'loss'
mode_stage1 = infer_mode_for_monitor(monitor_metric_stage1)
cb_stage1 = [
    callbacks.ReduceLROnPlateau(monitor=monitor_metric_stage1, factor=0.5, patience=3, verbose=1, min_lr=1e-8, mode=mode_stage1),
    callbacks.EarlyStopping(monitor=monitor_metric_stage1, patience=20, restore_best_weights=True, verbose=1, mode=mode_stage1)
]

compile_for_stage(cnn, lr=1e-2, loss_weights={'letter_output':1.0, 'combined_output':0.0, 'stage_output':0.0})
print("Starting Stage 1 training (letter-only)...")
if val_ds is not None:
    hist1 = cnn.fit(train_ds, validation_data=val_ds, epochs=stage1_epochs, callbacks=cb_stage1, verbose=1)
else:
    hist1 = cnn.fit(train_ds, epochs=stage1_epochs, callbacks=cb_stage1, verbose=1)

stage2_stage_loss_weight = 1.0 if stage_training_enabled else 0.0
compile_for_stage(cnn, lr=3e-3, loss_weights={'letter_output':1.0, 'combined_output':1.0, 'stage_output': stage2_stage_loss_weight})

monitor_metric_stage2 = 'val_loss' if val_ds is not None else 'loss'
mode_stage2 = infer_mode_for_monitor(monitor_metric_stage2)
cb_stage2 = [
    callbacks.ReduceLROnPlateau(monitor=monitor_metric_stage2, factor=0.25, patience=3, verbose=1, min_lr=1e-8, mode=mode_stage2),
    callbacks.EarlyStopping(monitor=monitor_metric_stage2, patience=20, restore_best_weights=True, verbose=1, mode=mode_stage2),
    callbacks.ModelCheckpoint("best_star_model_staged_combined.keras", monitor=monitor_metric_stage2, save_best_only=True, save_weights_only=False, verbose=1, mode=mode_stage2)
]

print("Starting Stage 2 training (multitask fine-tune)...")
if val_ds is not None:
    hist2 = cnn.fit(train_ds, validation_data=val_ds, epochs=stage2_epochs, callbacks=cb_stage2, verbose=1)
else:
    hist2 = cnn.fit(train_ds, epochs=stage2_epochs, callbacks=cb_stage2, verbose=1)

# =========================
# 19) Save Training History + Model
# =========================
history_dict = {}
try:
    if 'hist1' in locals() and hasattr(hist1, 'history'):
        history_dict.update({f"stage1_{k}": v for k,v in hist1.history.items()})
    if 'hist2' in locals() and hasattr(hist2, 'history'):
        history_dict.update({f"stage2_{k}": v for k,v in hist2.history.items()})
    if history_dict:
        with open("training_history_staged_combined.json","w") as fh: json.dump(history_dict, fh, indent=2)
        pd.DataFrame(history_dict).to_csv("training_history_staged_combined.csv", index=False)
        print("Saved staged training history.")
except Exception as e:
    print("Could not save staged training history:", e)

try:
    cnn.save("star_rgb_cnn_staged_combined.keras")
    print("Saved final staged model star_rgb_cnn_staged_combined.keras")
except Exception as e:
    print("Could not save final model:", e)

# =========================
# 20) NEW — Training Curves (Letters / Subclass / Stage / Overall)
# =========================
import numpy as np
import matplotlib.pyplot as plt

def _get_list(h, key):
    return list(h.get(key, [])) if h else []

def _concat_pad(a, b):
    return (a or []) + (b or [])

def _nan_like(n):
    return [np.nan] * n

def _plot_loss_acc(epochs, train_loss, val_loss, train_acc, val_acc, title, outfile):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(epochs, train_loss, label="Train Loss", linestyle="-")
    if any([x is not None and not np.isnan(x) for x in val_loss]):
        ax1.plot(epochs, val_loss, label="Val Loss", linestyle="--")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(epochs, train_acc, label="Train Acc", linestyle="-")
    if any([x is not None and not np.isnan(x) for x in val_acc]):
        ax2.plot(epochs, val_acc, label="Val Acc", linestyle="--")
    ax2.tick_params(axis="y")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(title)
    fig.tight_layout()
    try:
        plt.savefig(outfile, dpi=200)
        print(f"Saved plot -> {outfile}")
    except Exception as e:
        print("Could not save plot:", e)
    plt.show()

def _build_and_plot_all(hist1, hist2, stage1_epochs):
    h1 = hist1.history if hist1 is not None and hasattr(hist1, 'history') else {}
    h2 = hist2.history if hist2 is not None and hasattr(hist2, 'history') else {}
    n1 = len(next(iter(h1.values()))) if h1 else 0
    n2 = len(next(iter(h2.values()))) if h2 else 0
    epochs = list(range(1, n1 + n2 + 1))

    # Letters
    l_tr_loss = _concat_pad(_get_list(h1, "letter_output_loss"), _get_list(h2, "letter_output_loss"))
    l_va_loss = _concat_pad(_get_list(h1, "val_letter_output_loss"), _get_list(h2, "val_letter_output_loss"))
    l_tr_acc  = _concat_pad(_get_list(h1, "letter_output_accuracy"), _get_list(h2, "letter_output_accuracy"))
    l_va_acc  = _concat_pad(_get_list(h1, "val_letter_output_accuracy"), _get_list(h2, "val_letter_output_accuracy"))
    if not l_tr_acc:
        l_tr_acc = _concat_pad(_get_list(h1, "letter_output_acc"), _get_list(h2, "letter_output_acc"))
    if not l_va_acc:
        l_va_acc = _concat_pad(_get_list(h1, "val_letter_output_acc"), _get_list(h2, "val_letter_output_acc"))

    # Subclass (combined head)
    c_tr_loss = _concat_pad(_get_list(h1, "combined_output_loss"), _get_list(h2, "combined_output_loss"))
    c_va_loss = _concat_pad(_get_list(h1, "val_combined_output_loss"), _get_list(h2, "val_combined_output_loss"))
    c_tr_acc  = _concat_pad(_nan_like(n1), _get_list(h2, "combined_output_accuracy"))
    c_va_acc  = _concat_pad(_nan_like(n1), _get_list(h2, "val_combined_output_accuracy"))
    if not c_tr_acc or all(np.isnan(x) for x in c_tr_acc):
        c_tr_acc = _concat_pad(_nan_like(n1), _get_list(h2, "combined_output_acc"))
    if not c_va_acc or all(np.isnan(x) for x in c_va_acc):
        c_va_acc = _concat_pad(_nan_like(n1), _get_list(h2, "val_combined_output_acc"))

    # Stage
    s_tr_loss = _concat_pad(_get_list(h1, "stage_output_loss"), _get_list(h2, "stage_output_loss"))
    s_va_loss = _concat_pad(_get_list(h1, "val_stage_output_loss"), _get_list(h2, "val_stage_output_loss"))
    s_tr_acc  = _concat_pad(_nan_like(n1), _get_list(h2, "stage_output_accuracy"))
    s_va_acc  = _concat_pad(_nan_like(n1), _get_list(h2, "val_stage_output_accuracy"))
    if not s_tr_acc or all(np.isnan(x) for x in s_tr_acc):
        s_tr_acc = _concat_pad(_nan_like(n1), _get_list(h2, "stage_output_acc"))
    if not s_va_acc or all(np.isnan(x) for x in s_va_acc):
        s_va_acc = _concat_pad(_nan_like(n1), _get_list(h2, "val_stage_output_acc"))

    # Overall
    o_tr_loss = _concat_pad(_get_list(h1, "loss"), _get_list(h2, "loss"))
    o_va_loss = _concat_pad(_get_list(h1, "val_loss"), _get_list(h2, "val_loss"))

    def _avg_acc(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    overall_train_acc, overall_val_acc = [], []
    for i in range(n1 + n2):
        train_heads = []
        val_heads = []
        if i < len(l_tr_acc): train_heads.append(l_tr_acc[i])
        if i < len(c_tr_acc): train_heads.append(c_tr_acc[i])
        if i < len(s_tr_acc): train_heads.append(s_tr_acc[i])
        if i < len(l_va_acc): val_heads.append(l_va_acc[i])
        if i < len(c_va_acc): val_heads.append(c_va_acc[i])
        if i < len(s_va_acc): val_heads.append(s_va_acc[i])
        overall_train_acc.append(_avg_acc(train_heads))
        overall_val_acc.append(_avg_acc(val_heads))

    _plot_loss_acc(epochs, l_tr_loss, l_va_loss, l_tr_acc, l_va_acc,
                   "Letters — Loss & Accuracy over Epochs",
                   "plot_letters_loss_acc.png")

    _plot_loss_acc(epochs, c_tr_loss, c_va_loss, c_tr_acc, c_va_acc,
                   "Subclass (Combined Head) — Loss & Accuracy over Epochs",
                   "plot_subclass_combined_loss_acc.png")

    _plot_loss_acc(epochs, s_tr_loss, s_va_loss, s_tr_acc, s_va_acc,
                   "Stage — Loss & Accuracy over Epochs",
                   "plot_stage_loss_acc.png")

    _plot_loss_acc(epochs, o_tr_loss, o_va_loss, overall_train_acc, overall_val_acc,
                   "Overall — Loss & Avg Head Accuracy over Epochs",
                   "plot_overall_loss_acc.png")

try:
    _build_and_plot_all(hist1 if 'hist1' in locals() else None,
                        hist2 if 'hist2' in locals() else None,
                        stage1_epochs)
except Exception as e:
    print("History plotting failed:", e)

# =========================
# 21) Evaluation (test set + confusion matrices + NEW metrics)
# =========================
print("\nEvaluating on test set...")

# --- NEW: helper to compute top-k correctness given prob matrix ---
def _topk_correct(probs, true_labels, k=2):
    if probs is None or len(probs)==0 or len(true_labels)==0:
        return float('nan')
    topk = np.argpartition(-probs, kth=min(k-1, probs.shape[1]-1), axis=1)[:, :k]
    topk_sorted = np.take_along_axis(topk, np.argsort(-np.take_along_axis(probs, topk, axis=1), axis=1), axis=1)
    hits = np.any(topk_sorted == true_labels.reshape(-1,1), axis=1)
    return float(np.mean(hits)) if hits.size>0 else float('nan')

if len(X_test_norm) > 0:
    test_ds = make_dataset_with_scalar_sample_weight(X_test_norm, y_test_letter_cat, y_test_combined_cat, y_test_stage_cat, sw_test, batch_size=BATCH_SIZE, training=False)
    eval_res = cnn.evaluate(test_ds, verbose=1)
    print("Evaluate results (metrics vector):", eval_res)

    preds = cnn.predict(X_test_norm.astype('float32'), batch_size=BATCH_SIZE, verbose=1)
    pred_letters_probs = preds[0]
    pred_combined_probs = preds[1]
    pred_stage_probs = preds[2]

    pred_letters_enc = np.argmax(pred_letters_probs, axis=1)
    pred_combined_enc = np.argmax(pred_combined_probs, axis=1)
    pred_stage_enc = np.argmax(pred_stage_probs, axis=1)

    pred_letters_from_combined = pred_combined_enc // 10
    pred_subclass_from_combined = pred_combined_enc % 10

    letter_acc = float(np.mean(pred_letters_enc == y_test)) if len(y_test)>0 else float('nan')
    letter_acc_combined = float(np.mean(pred_letters_from_combined == y_test)) if len(y_test)>0 else float('nan')

    y_comb_true_test = (y_test * 10 + ysub_test).astype(int)
    combined_acc = float(np.mean(pred_combined_enc == y_comb_true_test)) if len(y_comb_true_test)>0 else float('nan')
    subclass_acc_overall = float(np.mean(pred_subclass_from_combined == ysub_test)) if len(ysub_test)>0 else float('nan')
    idx_letter_correct = np.where(pred_letters_enc == y_test)[0]
    if len(idx_letter_correct) > 0:
        subclass_acc_when_letter_correct = float(np.mean(pred_subclass_from_combined[idx_letter_correct] == ysub_test[idx_letter_correct]))
    else:
        subclass_acc_when_letter_correct = float('nan')

    stage_acc = float(np.mean(pred_stage_enc == ystage_test)) if len(ystage_test)>0 else float('nan')

    print(f"Test letter acc (letter head): {letter_acc:.4f}, letter acc (combined head): {letter_acc_combined:.4f}")
    print(f"Test combined (letter+subclass) acc: {combined_acc:.4f}")
    print(f"Subclass acc overall: {subclass_acc_overall:.4f}, when letter head correct: {subclass_acc_when_letter_correct:.4f}")
    print(f"Stage accuracy (stage head vs metadata-derived labels): {stage_acc:.4f}")

    # ---------- NEW: Top-2 and Macro P/R/F1 for Letters & Subclass ----------
    metrics_summary = {}

    # Letters (Top-1, Top-2, Macro P/R/F1) using letter head
    try:
        top1_letter = letter_acc
        top2_letter = _topk_correct(pred_letters_probs, y_test, k=2)
        prf_letter = precision_recall_fscore_support(y_test, pred_letters_enc, average='macro', zero_division=0)
        metrics_summary.update({
            "letters_top1_accuracy": float(top1_letter),
            "letters_top2_accuracy": float(top2_letter),
            "letters_macro_precision": float(prf_letter[0]),
            "letters_macro_recall": float(prf_letter[1]),
            "letters_macro_f1": float(prf_letter[2]),
        })
        print(f"\n[Letters] Top-1 Acc: {top1_letter:.4f} | Top-2 Acc: {top2_letter:.4f} | "
              f"Macro P/R/F1: {prf_letter[0]:.4f}/{prf_letter[1]:.4f}/{prf_letter[2]:.4f}")
    except Exception as e:
        print("Letter metrics failed:", e)

    # Subclass numbers via combined head:
    # - Top-1 is already subclass_acc_overall using argmax over combined head then %10
    # - Top-2: aggregate subclass probability by summing probs of all combined classes with the same subclass (mod 10)
    # - Macro P/R/F1 comparing true subclass vs predicted subclass_from_combined
    try:
        # aggregate subclass probabilities across letters
        if pred_combined_probs is not None and pred_combined_probs.size > 0:
            num_samples = pred_combined_probs.shape[0]
            subclass_probs = np.zeros((num_samples, 10), dtype=np.float32)
            for s in range(10):
                subclass_probs[:, s] = pred_combined_probs[:, s::10].sum(axis=1)
            top2_subclass = _topk_correct(subclass_probs, ysub_test, k=2)
        else:
            top2_subclass = float('nan')

        prf_sub = precision_recall_fscore_support(ysub_test, pred_subclass_from_combined, labels=list(range(10)), average='macro', zero_division=0)

        metrics_summary.update({
            "subclass_top1_accuracy": float(subclass_acc_overall),
            "subclass_top2_accuracy": float(top2_subclass),
            "subclass_macro_precision": float(prf_sub[0]),
            "subclass_macro_recall": float(prf_sub[1]),
            "subclass_macro_f1": float(prf_sub[2]),
            "combined_exact_accuracy": float(combined_acc)  # for convenience
        })
        print(f"[Subclass] Top-1 Acc: {subclass_acc_overall:.4f} | Top-2 Acc: {top2_subclass:.4f} | "
              f"Macro P/R/F1: {prf_sub[0]:.4f}/{prf_sub[1]:.4f}/{prf_sub[2]:.4f}")
        print(f"[Combined (letter+subclass)] Exact Acc: {combined_acc:.4f}")
    except Exception as e:
        print("Subclass metrics failed:", e)

    # Save a small CSV/JSON for easy reporting
    try:
        pd.DataFrame([metrics_summary]).to_csv("test_metrics_summary.csv", index=False)
        with open("test_metrics_summary.json","w") as f:
            json.dump(metrics_summary, f, indent=2)
        print("Saved metrics -> test_metrics_summary.csv / .json")
    except Exception as e:
        print("Could not write metrics summary files:", e)
    # ------------------------------------------------------------------------

    present_letters = list(le.classes_)
    ordered_present = [c for c in CANONICAL_LETTER_ORDER if c in present_letters]
    if len(ordered_present) == 0:
        order_indices = sorted(list(np.unique(y_test)))
    else:
        order_indices = [int(np.where(le.classes_ == c)[0][0]) for c in ordered_present]
    if len(order_indices) > 0 and len(y_test)>0:
        cm_eval = confusion_matrix(y_test, pred_letters_enc, labels=order_indices)
        plot_and_save_confusion(cm_eval, [le.classes_[i] for i in order_indices], 'Confusion Matrix (Letters - Test)', 'confusion_matrix_letters_test.png')

    for li, letter in enumerate(le.classes_):
        idxs = np.where(y_test == li)[0]
        if idxs.size == 0:
            continue
        true_subs = ysub_test[idxs]
        pred_subs = pred_subclass_from_combined[idxs]
        cm_sub = confusion_matrix(true_subs, pred_subs, labels=list(range(10)))
        plot_and_save_confusion(cm_sub, list(range(10)), f'Confusion Matrix (Subclass - Test) Letter {letter}', f'confusion_matrix_subclass_test_letter_{letter}.png')

    try:
        unique_stage_labels_test = sorted(list(np.unique(ystage_test)))
        meaningful_stage_labels_test = [s for s in unique_stage_labels_test if s != stage_to_idx['unknown']]
        if len(unique_stage_labels_test) > 1:
            cm_stage = confusion_matrix(ystage_test, pred_stage_enc, labels=unique_stage_labels_test)
            cm_stage_names = [STAGE_CLASSES[i] if i < len(STAGE_CLASSES) else str(i) for i in unique_stage_labels_test]
            plot_and_save_confusion(cm_stage, cm_stage_names, 'Confusion Matrix (Stellar Stage - Test)', 'confusion_matrix_stage_test.png')
    except Exception:
        pass

    try:
        unique_comb_labels = sorted(list(np.unique(y_comb_true_test)))
        if len(unique_comb_labels) > 1 and len(unique_comb_labels) <= 100:
            cm_comb = confusion_matrix(y_comb_true_test, pred_combined_enc, labels=unique_comb_labels)
            comb_labels_names = []
            for cb in unique_comb_labels:
                letter_idx = cb // 10
                sub_idx = cb % 10
                letter_name = le.classes_[letter_idx] if letter_idx < len(le.classes_) else str(letter_idx)
                comb_labels_names.append(f"{letter_name}_{sub_idx}")
            plot_and_save_confusion(cm_comb, comb_labels_names, 'Confusion Matrix (Combined class - Test)', 'confusion_matrix_combined_test.png')
    except Exception:
        pass

else:
    print("No test images to evaluate.")

# =========================
# 22) Evaluation vs Metadata Overlap (Optional)
# =========================
if meta_df is not None:
    print("\nEvaluating predictions vs metadata (if overlapping filenames present)...")
    meta_cols_lower = {c.lower():c for c in meta_df.columns}
    fname_candidates = [c for c in meta_df.columns if c.lower() in ('filename','file','raw_file','image','image_name','path','proc_filename')]
    spec_candidates = [c for c in meta_df.columns if c.lower() in ('spectral','spectral_type','spectraltype','spectral_class','spec','type','label','spectral_primary_letter')]
    if fname_candidates and spec_candidates:
        fname_col = fname_candidates[0]; spec_col = spec_candidates[0]
        image_lookup = build_image_lookup(filenames_final)
        metadata_map = {}; metadata_sub_map = {}
        for _, row in meta_df.iterrows():
            fn = row.get(fname_col,""); matched = try_match_meta_fname_to_image(fn, image_lookup)
            if not matched:
                for c in meta_df.columns:
                    if c == fname_col: continue
                    mtry = try_match_meta_fname_to_image(row.get(c,""), image_lookup)
                    if mtry: matched = mtry; break
            if not matched: continue
            lbl = row.get(spec_col,"")
            if pd.isna(lbl) or str(lbl).strip() == "": continue
            m = re.search(r'([OBAFGKM])', str(lbl).upper())
            if m: metadata_map[matched] = m.group(1)
            sub = None
            for sc in [c for c in meta_df.columns if c.lower() in ('spectral_subclass','subclass','sptype_subclass','spec_subclass')]:
                try:
                    sv = row.get(sc,None)
                    if sv is not None and not pd.isna(sv) and str(sv).strip() != "":
                        sub = int(round(float(sv))); break
                except Exception:
                    pass
            if sub is None:
                mm = re.search(r'([OBAFGKM])\s*([0-9])', str(lbl).upper())
                if mm:
                    try: sub = int(round(float(mm.group(2))))
                    except Exception: sub = None
            if sub is not None:
                metadata_sub_map[matched] = max(0,min(9,int(sub)))
        overlap_idx = [i for i,fn in enumerate(filenames_final) if fn in metadata_map]
        if overlap_idx:
            X_meta = np.array([X_final[i] for i in overlap_idx]).astype('float32')/255.0
            y_true_meta_letters = [metadata_map[filenames_final[i]] for i in overlap_idx]
            keep = [i for i,lab in enumerate(y_true_meta_letters) if lab in le.classes_]
            if not keep:
                print("Metadata overlap had no labels in our classes.")
            else:
                idx_keep_global = [overlap_idx[i] for i in keep]
                X_meta_keep = np.array([X_final[i] for i in idx_keep_global]).astype('float32')/255.0
                y_true_enc_letters = le.transform([metadata_map[filenames_final[i]] for i in idx_keep_global])
                preds_meta = cnn.predict(X_meta_keep)
                pred_letters_meta = np.argmax(preds_meta[0], axis=1)
                pred_combined_meta = np.argmax(preds_meta[1], axis=1)
                pred_letters_meta_from_combined = pred_combined_meta // 10
                pred_sub_meta_from_combined = pred_combined_meta % 10
                letter_acc_meta = float(np.mean(pred_letters_meta == y_true_enc_letters))
                letter_acc_meta_comb = float(np.mean(pred_letters_meta_from_combined == y_true_enc_letters))
                print("Accuracy on metadata-overlap subset (letters):", letter_acc_meta, " (combined head):", letter_acc_meta_comb)
                order_idx_meta = sorted(list(np.unique(np.concatenate([y_true_enc_letters, pred_letters_meta]))))
                cm_meta = confusion_matrix(y_true_enc_letters, pred_letters_meta, labels=order_idx_meta)
                plot_and_save_confusion(cm_meta, [le.classes_[i] for i in order_idx_meta], 'Confusion Matrix (Eval vs Metadata - letters)', 'confusion_matrix_eval_vs_metadata_letters.png')
        else:
            print("No overlap between filenames and metadata.")
    else:
        print("Metadata present but couldn't identify filename & spectral columns for evaluation.")
else:
    print("No metadata to evaluate against.")

# =========================
# 23) Save Final Derived Labels
# =========================
if SAVE_DERIVED_LABELS:
    try:
        df_final = pd.DataFrame({
            "filename":filenames_final,
            "label_used_for_training":labels_final,
            "estimated_CCT":cct_final,
            "subclass_used_for_training":subclass_final,
            "subclass_confidence":subclass_conf_final
        })
        df_final['stage_from_metadata'] = [STAGE_CLASSES[i] for i in y_stage_all]
        df_final.to_csv("derived_labels_used_for_training_balanced_curved_multitask_final.csv", index=False)
        print("Saved final derived labels CSV.")
    except Exception as e:
        print("Could not save final derived labels CSV:", e)

print("Script finished.")

### Wasnt able to get the Stellar stage up and running in time so it should jsut be outputting 0 accuracy presently
### "combined" refers to Letter+number classes , stage is not taken into account for that one
### zip file MUST be renamed to star_images_bundle.zip
