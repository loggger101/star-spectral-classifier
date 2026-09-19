#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================
# Stellar CNN Inference (Colab single cell) — robust letter mapping & ensemble decoding
# ============================

# --- (edit these paths) ---
MODEL_PATH = "star_rgb_cnn_staged_combined.keras"   # your saved model (.keras/.h5/SavedModel dir)
INPUT_PATH = "one_image_or_folder"           # a file path OR a folder path
GROUND_TRUTH_CSV = "metadata.csv"                   # NEEDED if you want proper classes
CLASSES_JSON_PATH = None                  # optional but recommended; set None if you don't have it.

# Inference behavior
STAGE_DOMINANCE_SUPPRESS = True   # if stage head collapses to 1 class, report 'unknown'
DOMINANCE_THRESHOLD = 0.90        # stage dominance fraction to trigger suppression
LETTER_BLEND = 0.5                # 0..1; 0=only combined head, 1=only letter head; 0.5 uses both
AUTO_REMAP_LETTERS = True         # learn a permutation (pred_idx -> real letter) from GT if available
SHOW_IMAGES = True                # set False to skip matplotlib displays


# ============================
# Imports
# ============================
import os, glob, re, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# ============================
# Helpers
# ============================
def softmax(x, axis=-1, eps=1e-8):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    s = e / (np.sum(e, axis=axis, keepdims=True) + eps)
    return s

def collect_images(path):
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "*.*")))
        paths = [p for p in paths if p.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
    elif os.path.isfile(path):
        paths = [path]
    else:
        raise FileNotFoundError(f"INPUT_PATH not found: {path}")
    return paths

def parse_from_filename(fname_no_ext):
    # Match patterns like G2V / K1III / F5 / or with canonical words G2_main_sequence
    m = re.search(
        r'([OBAFGKM])(\d)(?:_(white_dwarf|main_sequence|subgiant|giant|supergiant|unknown)|(I{1,3}|IV|V))?$',
        fname_no_ext, re.IGNORECASE)
    if not m:
        return None
    letter = m.group(1).upper()
    subclass = m.group(2)
    stage_txt = m.group(3) or m.group(4)
    if stage_txt and stage_txt.upper() in ('I','II','III','IV','V'):
        stage = stage_txt.upper()
    elif stage_txt:
        stage = stage_txt.lower()
    else:
        stage = None
    return letter + subclass + (stage if stage else "")

def load_ground_truth(csv_path):
    """
    Tries common column names used in your training pipeline.
    Returns dict: {basename -> "L#stage" or "L#"} plus set of letters seen.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    # filename column
    fname_cols = [c for c in df.columns if c.lower() in (
        'filename','file','raw_file','image','image_name','proc_filename','path')]
    if not fname_cols:
        return {}, set()
    fname_col = fname_cols[0]
    # letter column
    letter_col = None
    for cand in ('spectral_primary_letter','spectral_type_letter','spectral_type',
                 'spectral','spectralclass','spectral_class','spectraltype','spec','type','label'):
        if cand in cols:
            letter_col = cols[cand]; break
    # subclass column
    subclass_col = None
    for cand in ('spectral_subclass','subclass','spec_subclass','sptype_subclass'):
        if cand in cols:
            subclass_col = cols[cand]; break
    # stage column
    stage_col = None
    for cand in ('stellar_stage','stage','lumclass','luminosity_class',
                 'spectral_luminosity','luminosity','class'):
        if cand in cols:
            stage_col = cols[cand]; break

    def canon_stage(s):
        if s is None or (isinstance(s,float) and np.isnan(s)): return None
        ss = str(s).strip().lower()
        if ss == "": return None
        if 'white' in ss or 'wd' in ss or 'dwarf' in ss: return 'white_dwarf'
        if 'super' in ss or ss.startswith('i '):        return 'supergiant'
        if 'iii' in ss or ' giant' in ss or ss=='iii':  return 'giant'
        if 'iv' in ss or ss=='iv':                      return 'subgiant'
        if 'main' in ss or ss=='v' or ss.startswith('v'): return 'main_sequence'
        return None

    GT = {}
    letters_seen = set()
    for _, row in df.iterrows():
        fn = os.path.basename(str(row.get(fname_col, "")).strip())
        if not fn: continue
        letter = None
        if letter_col:
            m = re.search(r'([OBAFGKM])', str(row.get(letter_col, "")).upper())
            if m: letter = m.group(1)
        subclass = None
        if subclass_col:
            try:
                sv = row.get(subclass_col, None)
                if sv is not None and str(sv).strip() != "":
                    subclass = str(int(round(float(sv))))
            except Exception:
                pass
        stage = None
        if stage_col:
            stage = canon_stage(row.get(stage_col, None))
        if letter and subclass is not None:
            lab = letter + subclass + (stage if stage else "")
            GT[fn] = lab
            letters_seen.add(letter)
    return GT, letters_seen

def greedy_assign_max(conf_mat):
    """
    Greedy one-to-one assignment that maximizes matches:
    conf_mat[pred_idx, actual_idx] = counts
    Returns mapping list of length num_pred: pred_idx -> actual_idx (or None).
    """
    num_pred, num_actual = conf_mat.shape
    mapping = [-1]*num_pred
    used_actual = set()
    # Flatten and sort by counts descending
    pairs = [(i,j,conf_mat[i,j]) for i in range(num_pred) for j in range(num_actual)]
    pairs.sort(key=lambda x: -x[2])
    for i,j,c in pairs:
        if c <= 0: break
        if mapping[i] == -1 and j not in used_actual:
            mapping[i] = j
            used_actual.add(j)
    return mapping

# ============================
# Load model
# ============================
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")
try:
    print("Model outputs:", model.output_names)
except Exception:
    pass

# ============================
# Classes / defaults
# ============================
CANONICAL_LETTERS = ['O','B','A','F','G','K','M']
DEFAULT_STAGE_CLASSES = ['white_dwarf','main_sequence','subgiant','giant','supergiant','unknown']
DEFAULT_SUBCLASSES = [str(i) for i in range(10)]  # 0..9

letter_classes = None
stage_classes = None
subclass_classes = DEFAULT_SUBCLASSES

# Try classes.json
classes_json = None
if CLASSES_JSON_PATH and os.path.isfile(CLASSES_JSON_PATH):
    try:
        with open(CLASSES_JSON_PATH, "r") as f:
            classes_json = json.load(f)
        print("Loaded classes.json:", classes_json)
    except Exception as e:
        print("Could not parse classes.json:", e)

if classes_json:
    letter_classes = classes_json.get("letter_classes", None)
    stage_classes  = classes_json.get("stage_classes", None)

if stage_classes is None:
    stage_classes = DEFAULT_STAGE_CLASSES

# ============================
# Collect & preprocess images
# ============================
image_paths = collect_images(INPUT_PATH)
if len(image_paths) == 0:
    raise RuntimeError("No images found. Check INPUT_PATH.")
print(f"Found {len(image_paths)} image(s).")

# Determine input size
try:
    target_h = model.input_shape[1] or 64
    target_w = model.input_shape[2] or 64
except Exception:
    target_h, target_w = 64, 64

def preprocess_image(p):
    img = load_img(p, target_size=(target_h, target_w))
    arr = img_to_array(img).astype('float32') / 255.0
    return arr

images = np.stack([preprocess_image(p) for p in image_paths], axis=0)
print(f"Preprocessed batch shape: {images.shape}")

# ============================
# Ground truth: CSV + filename parse
# ============================
GT = {}
letters_from_gt = set()
if GROUND_TRUTH_CSV and os.path.isfile(GROUND_TRUTH_CSV):
    try:
        GT_csv, letters_csv = load_ground_truth(GROUND_TRUTH_CSV)
        GT.update(GT_csv)
        letters_from_gt |= letters_csv
        print(f"Loaded {len(GT_csv)} ground-truth rows from CSV.")
    except Exception as e:
        print("Failed to load ground-truth CSV:", e)

# Add filename-parsed GT for the current batch
for p in image_paths:
    base = os.path.basename(p)
    if base not in GT:
        parsed = parse_from_filename(os.path.splitext(base)[0])
        if parsed:
            GT[base] = parsed
            letters_from_gt.add(parsed[0])

# ============================
# Predict
# ============================
raw_preds = model.predict(images, batch_size=32, verbose=0)

# Normalize to dict {output_name: np.ndarray}
if isinstance(raw_preds, list):
    outnames = getattr(model, "output_names", None)
    if outnames and len(outnames)==len(raw_preds):
        preds = {outnames[i]: raw_preds[i] for i in range(len(raw_preds))}
    else:
        preds = {f"out_{i}": raw_preds[i] for i in range(len(raw_preds))}
elif isinstance(raw_preds, dict):
    preds = raw_preds
else:
    raise RuntimeError("Unexpected single-output model; expected 3 outputs.")

# Identify heads: prefer explicit names, else fallback by shape heuristics
def find_heads(preds_dict, stage_len):
    sizes = {k: v.shape[-1] for k,v in preds_dict.items()}
    letter_key   = "letter_output"   if "letter_output"   in preds_dict else None
    combined_key = "combined_output" if "combined_output" in preds_dict else None
    stage_key    = "stage_output"    if "stage_output"    in preds_dict else None
    if letter_key and combined_key and stage_key:
        return letter_key, combined_key, stage_key
    # heuristics
    if letter_key is None:
        for k,s in sizes.items():
            if 2 <= s <= 7: letter_key = k; break
    if combined_key is None:
        for k,s in sizes.items():
            if s % 10 == 0 and s >= 20: combined_key = k; break
    if stage_key is None:
        for k,s in sizes.items():
            if s == stage_len: stage_key = k; break
    keys = list(preds_dict.keys())
    if letter_key   is None: letter_key   = keys[0]
    if combined_key is None: combined_key = keys[1 if len(keys)>1 else 0]
    if stage_key    is None: stage_key    = keys[2 if len(keys)>2 else -1]
    return letter_key, combined_key, stage_key

letter_key, combined_key, stage_key = find_heads(preds, len(stage_classes))
P_letter = preds[letter_key]    # [N, L]
P_comb   = preds[combined_key]  # [N, L*10]
P_stage  = preds[stage_key]     # [N, S]

# Letter classes length
num_letters = P_letter.shape[-1]
if (letter_classes is None) or (len(letter_classes) != num_letters):
    # must infer. Training used LabelEncoder on present letters (alphabetical subset).
    # Without saved order, will learn a permutation from GT if possible.
    print(f"[info] No letter_classes or size mismatch; will infer mapping from GT if available.")
    # provisional names just placeholders
    letter_classes = [f"IDX{j}" for j in range(num_letters)]

# Stage sanity: detect collapse (dominance of a single class)
top_stage = np.argmax(P_stage, axis=1)
vals, cnt = np.unique(top_stage, return_counts=True)
dom_frac = cnt.max()/cnt.sum()
force_unknown_stage = False
if dom_frac > DOMINANCE_THRESHOLD:
    print(f"[warn] Stage head heavily biased: class index {int(vals[cnt.argmax()])} covers {dom_frac:.1%} of samples.")
    if classes_json and not classes_json.get("stage_training_enabled", True):
        print("[hint] Stage head likely NOT trained (loss weight 0).")
    if STAGE_DOMINANCE_SUPPRESS:
        force_unknown_stage = True

# Combined head reshape
if P_comb.shape[-1] % 10 != 0:
    raise RuntimeError(f"Combined head size {P_comb.shape[-1]} is not a multiple of 10.")
comb_letters = P_comb.shape[-1] // 10
if comb_letters != num_letters:
    print(f"[warn] combined head implies {comb_letters} letters, but letter head has {num_letters}. Using letter head count.")
P_comb_3d = P_comb.reshape((-1, comb_letters, 10))

# ============================
# Build per-letter probabilities & optional auto-remap from GT
# ============================
# Per-letter from combined head (sum subclasses)
P_letter_from_comb = P_comb_3d.sum(axis=2)  # [N, L]

# Blend scores for final letter choice
P_letter_norm = softmax(P_letter, axis=1)
P_letter_from_comb_norm = softmax(P_letter_from_comb, axis=1)
blend = np.clip(float(LETTER_BLEND), 0.0, 1.0)
P_letter_blend = blend * P_letter_norm + (1.0 - blend) * P_letter_from_comb_norm  # [N, L]

# If LETTER head looks collapsed, auto-switch to combined-only
tops = np.argmax(P_letter_norm, axis=1)
valsL, cntL = np.unique(tops, return_counts=True)
if cntL.max()/cntL.sum() > 0.95 and LETTER_BLEND > 0.0:
    print("[info] Letter head looks collapsed; switching to combined-only letter decoding.")
    P_letter_blend = P_letter_from_comb_norm

# Try to learn mapping pred_idx -> real letter from GT (greedy 1-1 assignment)
remap = None
if AUTO_REMAP_LETTERS and len(GT) > 0 and len(letters_from_gt) > 0:
    # Actual letters present in GT; sort alphabetically (LabelEncoder order)
    actual_letters_sorted = sorted(list(letters_from_gt))[:num_letters]  # cap to L
    actual_index = {L:i for i,L in enumerate(actual_letters_sorted)}
    conf = np.zeros((num_letters, len(actual_letters_sorted)), dtype=np.int64)  # [pred_idx, actual_idx]
    # Build confusion using current blended top-1 indices
    for idx, p in enumerate(image_paths):
        base = os.path.basename(p)
        gt = GT.get(base, None)
        if not gt or len(gt) < 2: continue
        gt_letter = gt[0].upper()
        if gt_letter not in actual_index: continue
        pred_idx = int(np.argmax(P_letter_blend[idx]))
        conf[pred_idx, actual_index[gt_letter]] += 1
    if conf.sum() > 0:
        mapping = greedy_assign_max(conf)  # list length L, pred_idx -> actual_idx or -1
        # Build remap: pred_idx -> real letter string
        remap = {}
        for pred_idx, act_idx in enumerate(mapping):
            if act_idx != -1:
                remap[pred_idx] = actual_letters_sorted[act_idx]
        # Fill any leftovers with the most frequent actual letters not used yet
        used = set(remap.values())
        leftovers = [a for a in actual_letters_sorted if a not in used]
        for pred_idx in range(num_letters):
            if pred_idx not in remap:
                remap[pred_idx] = leftovers.pop(0) if leftovers else actual_letters_sorted[-1]
        print("[info] Learned letter index -> real letter mapping from GT:", remap)
        # Replace placeholder letter_classes with mapped real letters in index order
        letter_classes = [remap[i] for i in range(num_letters)]
    else:
        print("[info] Not enough GT overlap to learn letter mapping. Using blended decoding with placeholder names.")

# ============================
# Decode predictions
# ============================
results = []  # (image_path, pred_str, gt_str_or_None)

for i, path in enumerate(image_paths):
    # 1) Letter from blended scores
    li = int(np.argmax(P_letter_blend[i]))
    letter = letter_classes[li] if li < len(letter_classes) else f"IDX{li}"

    # 2) Subclass from combined head restricted to the chosen letter
    li_comb = li if li < P_comb_3d.shape[1] else 0
    subclass_i = int(np.argmax(P_comb_3d[i, li_comb, :]))
    subclass = DEFAULT_SUBCLASSES[subclass_i] if subclass_i < len(DEFAULT_SUBCLASSES) else '?'

    # 3) Stage from stage head (or 'unknown' if suppressed)
    si = int(np.argmax(P_stage[i]))
    stage_decoded = stage_classes[si] if si < len(stage_classes) else 'unknown'
    stage = 'unknown' if force_unknown_stage else stage_decoded

    pred_str = f"{letter}{subclass}"

    # Ground truth
    base = os.path.basename(path)
    gt = GT.get(base, None)

    # Print result line
    if gt:
        print(f"{base}: Predicted={pred_str} | Actual={gt}")
    else:
        print(f"{base}: Predicted={pred_str}")

    results.append((path, pred_str, gt))

# ============================
# Visualize (matplotlib)
# ============================
def show_with_text(img_arr, pred, gt=None):
    plt.imshow(img_arr)
    txt = f"Predicted: {pred}" + (f"\nActual: {gt}" if gt else "")
    plt.text(5, 16, txt, color='white', fontsize=12,
             bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    plt.axis('off')
    plt.show()

if SHOW_IMAGES:
    for (path, pred, gt) in results:
        idx = image_paths.index(path)
        show_with_text(images[idx], pred, gt)

print("✅ Done.")
