#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-builder + training-data exporter (no CNN) — optimized + optional PyTorch GPU paths

Fix: workers return bytes and main thread writes sequential filenames under lock,
so saved image numbering is contiguous and no leftover worker-indexed files remain.
Other behavior preserved.
"""
import os
import sys
import math
import argparse
import shutil
import zipfile
import json
import numpy as np
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from PIL import Image, ImageFilter
from astropy.table import Table
from astroquery.vizier import Vizier
import colorsys
import re
import warnings
from collections import Counter
import threading
import concurrent.futures
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional torch import for GPU acceleration/benchmarking
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

Vizier.ROW_LIMIT = -1

# -----------------------------
# Requests session & helpers (persistent + retries)
# -----------------------------
_global_session = None
_session_lock = threading.Lock()

def init_requests_session(timeout=15, max_retries=3, backoff_factor=0.3):
    global _global_session
    with _session_lock:
        if _global_session is None:
            sess = requests.Session()
            retries = Retry(
                total=max_retries,
                backoff_factor=backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(['GET', 'POST'])
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
            sess.mount('https://', adapter)
            sess.mount('http://', adapter)
            sess.headers.update({'User-Agent': 'star-image-builder/1.0'})
            _global_session = sess
    return _global_session

def get_session_or_create(timeout=15):
    return init_requests_session(timeout=timeout)

# -----------------------------
# PyTorch GPU benchmarking
# -----------------------------
def timed_forward_torch(device, iters=500, n=10000, d=1024, k=1024, dtype=torch.float32):
    x = torch.randn(n, d, device=device, dtype=(torch.float32 if device.type == "cpu" else dtype))
    w = torch.randn(d, k, device=device, dtype=(torch.float32 if device.type == "cpu" else dtype))
    b = torch.randn(k, device=device, dtype=(torch.float32 if device.type == "cpu" else dtype))
    for _ in range(10):
        _ = (x @ w + b).relu()
        if device.type == "cuda":
            torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = (x @ w + b).relu()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / iters

def run_gpu_benchmark(iters=500, n=10000, d=1024, k=1024):
    if not _HAS_TORCH:
        print("PyTorch not installed; cannot run GPU benchmark.")
        return None
    cpu_t = timed_forward_torch(torch.device("cpu"), iters=iters, n=n, d=d, k=k, dtype=torch.float32)
    print(f"CPU fp32 avg: {cpu_t*1000:.2f} ms")
    if torch.cuda.is_available():
        try:
            gpu_t = timed_forward_torch(torch.device("cuda"), iters=iters, n=n, d=d, k=k, dtype=torch.float32)
            print(f"GPU fp32 avg: {gpu_t*1000:.2f} ms    Speedup x{cpu_t/max(gpu_t,1e-9):.1f}")
            return {"cpu_ms": cpu_t*1000.0, "gpu_ms": gpu_t*1000.0, "speedup": cpu_t/max(gpu_t,1e-9)}
        except Exception as e:
            print("GPU benchmark failed:", e)
            return {"cpu_ms": cpu_t*1000.0}
    else:
        print("GPU not available.")
        return {"cpu_ms": cpu_t*1000.0}

# -----------------------------
# PS1 helpers & skyview (use session)
# -----------------------------
def ps1_filenames_for_position(ra, dec, filters="gri"):
    url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters={filters}"
    try:
        t = Table.read(url, format="ascii")
        return t
    except Exception:
        return None

def get_ps1_cutout_from_filenames(ra, dec, size_px=256, filters="gri", timeout=20):
    sess = get_session_or_create(timeout)
    try:
        t = ps1_filenames_for_position(ra, dec, filters=filters)
        if t is None or len(t) == 0:
            return None
        g_row = next((r for r in t if str(r.get("filter", "")).lower() == "g"), None)
        r_row = next((r for r in t if str(r.get("filter", "")).lower() == "r"), None)
        i_row = next((r for r in t if str(r.get("filter", "")).lower() == "i"), None)
        if not (g_row and r_row and i_row):
            return None
        fname_g = g_row["filename"]; fname_r = r_row["filename"]; fname_i = i_row["filename"]
        base = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
        params = {
            "red": fname_i, "green": fname_r, "blue": fname_g,
            "format": "png", "output_size": str(size_px), "size": str(size_px),
            "ra": str(ra), "dec": str(dec)
        }
        params_str = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in params.items())
        url = f"{base}?{params_str}"
        resp = sess.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        return None
    return None

def get_ps1_cutout_simple(ra, dec, size_px=256, timeout=15):
    sess = get_session_or_create(timeout)
    url = (
        f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        f"ra={ra}&dec={dec}&size={size_px}&format=png&output_size={size_px}"
        f"&red=i&green=r&blue=g"
    )
    try:
        r = sess.get(url, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        return None
    return None

def get_skyview_cutout(ra, dec, size_px=256, survey="DSS2 Red", timeout=15):
    sess = get_session_or_create(timeout)
    url = (
        "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
        f"?Position={ra},{dec}&Survey={survey}&Size=0.008&Pixels={size_px},{size_px}&Return=PNG"
    )
    try:
        r = sess.get(url, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        return None
    return None

# -----------------------------
# Heuristic grayscale detection
# -----------------------------
def is_effectively_grayscale(img_bytes, threshold_mean_diff=2.5):
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
        arr = np.array(im).astype(float)
        ch0 = arr[:, :, 0]; ch1 = arr[:, :, 1]; ch2 = arr[:, :, 2]
        mad = (np.abs(ch0 - ch1) + np.abs(ch0 - ch2) + np.abs(ch1 - ch2)) / 3.0
        mean_mad = np.nanmean(mad)
        return float(mean_mad) < threshold_mean_diff
    except Exception:
        return False

# -----------------------------
# Color normalization helper (enforce plausible star colors)
# -----------------------------
def normalize_star_colors(arr_uint8, teff=None):
    try:
        arr = arr_uint8.astype(np.float32)
        orig_luma = 0.2126 * arr[:, :, 0].mean() + 0.7152 * arr[:, :, 1].mean() + 0.0722 * arr[:, :, 2].mean()
        avg = arr.mean(axis=(0, 1))
        r_mean, g_mean, b_mean = float(avg[0]), float(avg[1]), float(avg[2])
        small_eps = 1e-6
        mx_rb = max(r_mean, b_mean, small_eps)
        if g_mean > mx_rb * 1.20:
            target_g_mean = (r_mean + b_mean) / 2.0
            scale = max(0.5, min(1.0, target_g_mean / (g_mean + small_eps)))
            arr[:, :, 1] *= scale
        if teff is not None:
            try:
                t = float(teff)
                t = max(2400.0, min(50000.0, t))
                frac = (t - 2400.0) / (50000.0 - 2400.0)
                red_fac = 1.0 + 0.22 * (1.0 - frac)
                blue_fac = 1.0 + 0.22 * frac
                arr[:, :, 0] *= red_fac
                arr[:, :, 2] *= blue_fac
            except Exception:
                pass
        new_luma = 0.2126 * arr[:, :, 0].mean() + 0.7152 * arr[:, :, 1].mean() + 0.0722 * arr[:, :, 2].mean()
        if new_luma > 0 and orig_luma > 0:
            scale_luma = orig_luma / new_luma
            scale_luma = max(0.7, min(1.4, scale_luma))
            arr *= scale_luma
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        avg2 = arr.mean(axis=(0,1)).astype(float)
        if float(avg2[1]) > max(float(avg2[0]), float(avg2[2])) * 1.25:
            arr = arr.astype(np.float32)
            g_scale = (max(avg2[0], avg2[2]) * 1.0) / (avg2[1] + small_eps)
            g_scale = max(0.6, min(1.0, g_scale))
            arr[:, :, 1] *= g_scale
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    except Exception:
        try:
            return arr_uint8.astype(np.uint8)
        except Exception:
            return np.zeros_like(arr_uint8, dtype=np.uint8)

# -----------------------------
# Strong background removal processor (with optional GPU path)
# -----------------------------
def process_strong_background(img_bytes, size_px=256, annulus_r_in=10, annulus_r_out=30,
                              lowfreq_blur=10, target_peak=180.0, peak_clamp=(0.4, 6.0),
                              contrast_percentiles=(1, 99), unsharp_radius=1.0, median_denoise_size=1,
                              use_gpu=False, gpu_device=None):
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None, None, None
    if im.size != (size_px, size_px):
        im = im.resize((size_px, size_px), resample=Image.LANCZOS)
    arr = np.array(im).astype(float)
    h, w = arr.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if use_gpu and _HAS_TORCH and torch.cuda.is_available():
        try:
            device = gpu_device or torch.device("cuda")
            t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            t_r = torch.from_numpy(r).to(device=device, dtype=torch.float32)
            mask_annulus = (t_r >= float(annulus_r_in)) & (t_r <= float(annulus_r_out))
            bg_vals = []
            for ch in range(3):
                chvals = t[:, :, ch][mask_annulus]
                if chvals.numel() == 0:
                    med = 0.0
                else:
                    try:
                        lo = torch.quantile(chvals, 0.05)
                        hi = torch.quantile(chvals, 0.95)
                        trimmed = chvals[(chvals >= lo) & (chvals <= hi)]
                        med = float(torch.median(trimmed).item()) if trimmed.numel() else float(torch.median(chvals).item())
                    except Exception:
                        vals_cpu = chvals.detach().cpu().numpy()
                        if vals_cpu.size == 0:
                            med = 0.0
                        else:
                            lo_np, hi_np = np.percentile(vals_cpu, [5, 95])
                            trimmed = vals_cpu[(vals_cpu >= lo_np) & (vals_cpu <= hi_np)]
                            med = float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(vals_cpu))
                bg_vals.append(med)
            bg = torch.tensor(bg_vals, device=device, dtype=torch.float32).view(1, 1, 3)
            t_sub = t - bg
            t_sub = torch.clamp(t_sub, min=0.0)
            arr_sub_cpu = t_sub.detach().cpu().numpy().astype(np.uint8)
            pil_temp = Image.fromarray(np.clip(arr_sub_cpu, 0, 255).astype(np.uint8))
            lowfreq = pil_temp.filter(ImageFilter.GaussianBlur(radius=lowfreq_blur))
            lowfreq_arr = np.array(lowfreq).astype(float)
            arr_sub2 = arr_sub_cpu - lowfreq_arr
            arr_sub2 = np.where(arr_sub2 < 0, 0.0, arr_sub2)
            p_lo, p_hi = contrast_percentiles
            stretched = np.zeros_like(arr_sub2)
            try:
                t_arr = torch.from_numpy(arr_sub2).to(device=device, dtype=torch.float32)
                for ch in range(3):
                    chvals = t_arr[:, :, ch].flatten()
                    if chvals.numel() == 0:
                        stretched[:, :, ch] = arr_sub2[:, :, ch]
                        continue
                    lo = float(torch.quantile(chvals, p_lo / 100.0).item())
                    hi = float(torch.quantile(chvals, p_hi / 100.0).item())
                    if hi <= lo:
                        stretched[:, :, ch] = arr_sub2[:, :, ch]
                    else:
                        stretched[:, :, ch] = (arr_sub2[:, :, ch] - lo) / (hi - lo) * 255.0
            except Exception:
                for ch in range(3):
                    chvals = arr_sub2[:, :, ch].flatten()
                    if chvals.size == 0:
                        stretched[:, :, ch] = arr_sub2[:, :, ch]
                        continue
                    lo = np.percentile(chvals, p_lo)
                    hi = np.percentile(chvals, p_hi)
                    if hi <= lo:
                        stretched[:, :, ch] = arr_sub2[:, :, ch]
                    else:
                        stretched[:, :, ch] = (arr_sub2[:, :, ch] - lo) / (hi - lo) * 255.0
            stretched = np.clip(stretched, 0, 255)
            half = 2
            y0 = int(round(cy)) - half; y1 = y0 + 2 * half + 1
            x0 = int(round(cx)) - half; x1 = x0 + 2 * half + 1
            y0, y1 = max(0, y0), min(h, y1)
            x0, x1 = max(0, x0), min(w, x1)
            central_patch = stretched[y0:y1, x0:x1, :]
            current_peak = float(central_patch.max()) if central_patch.size > 0 else float(stretched.max())
            if current_peak <= 0 or target_peak <= 0:
                scale = 1.0
            else:
                raw_scale = float(target_peak) / float(current_peak)
                mn, mx = peak_clamp
                scale = max(mn, min(mx, raw_scale))
            final = np.clip(stretched * scale, 0, 255).astype(np.uint8)
            pil_final = Image.fromarray(final)
            if unsharp_radius and unsharp_radius > 0:
                pil_final = pil_final.filter(ImageFilter.UnsharpMask(radius=unsharp_radius, percent=120, threshold=3))
            if median_denoise_size and median_denoise_size > 0:
                pil_final = pil_final.filter(ImageFilter.MedianFilter(size=max(1, median_denoise_size)))
            out_buf = BytesIO()
            pil_final.convert("RGB").save(out_buf, format="PNG")
            return out_buf.getvalue(), current_peak, scale
        except Exception:
            pass

    mask_annulus = (r >= annulus_r_in) & (r <= annulus_r_out)
    if mask_annulus.sum() < 50:
        mask_annulus = r >= min(annulus_r_in, int(min(h, w) / 4))

    bg_vals = []
    for ch in range(3):
        vals = arr[:, :, ch][mask_annulus].flatten()
        if vals.size == 0:
            med = 0.0
        else:
            lo, hi = np.percentile(vals, [5, 95])
            trimmed = vals[(vals >= lo) & (vals <= hi)]
            med = float(np.median(trimmed)) if trimmed.size > 0 else float(np.median(vals))
        bg_vals.append(med)
    bg = np.array(bg_vals).reshape((1, 1, 3))

    arr_sub = arr - bg
    arr_sub = np.where(arr_sub < 0, 0.0, arr_sub)
    try:
        pil_temp = Image.fromarray(np.clip(arr_sub, 0, 255).astype(np.uint8))
        lowfreq = pil_temp.filter(ImageFilter.GaussianBlur(radius=lowfreq_blur))
        lowfreq_arr = np.array(lowfreq).astype(float)
        arr_sub2 = arr_sub - lowfreq_arr
    except Exception:
        arr_sub2 = arr_sub
    arr_sub2 = np.where(arr_sub2 < 0, 0.0, arr_sub2)
    p_lo, p_hi = contrast_percentiles
    stretched = np.zeros_like(arr_sub2)
    for ch in range(3):
        chvals = arr_sub2[:, :, ch].flatten()
        if chvals.size == 0:
            stretched[:, :, ch] = arr_sub2[:, :, ch]
            continue
        lo = np.percentile(chvals, p_lo)
        hi = np.percentile(chvals, p_hi)
        if hi <= lo:
            stretched[:, :, ch] = arr_sub2[:, :, ch]
        else:
            stretched[:, :, ch] = (arr_sub2[:, :, ch] - lo) / (hi - lo) * 255.0
    stretched = np.clip(stretched, 0, 255)
    half = 2
    y0 = int(round(cy)) - half; y1 = y0 + 2 * half + 1
    x0 = int(round(cx)) - half; x1 = x0 + 2 * half + 1
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    central_patch = stretched[y0:y1, x0:x1, :]
    current_peak = float(central_patch.max()) if central_patch.size > 0 else float(stretched.max())
    if current_peak <= 0 or target_peak <= 0:
        scale = 1.0
    else:
        raw_scale = float(target_peak) / float(current_peak)
        mn, mx = peak_clamp
        scale = max(mn, min(mx, raw_scale))
    final = np.clip(stretched * scale, 0, 255).astype(np.uint8)
    try:
        pil_final = Image.fromarray(final)
        if unsharp_radius and unsharp_radius > 0:
            pil_final = pil_final.filter(ImageFilter.UnsharpMask(radius=unsharp_radius, percent=120, threshold=3))
        if median_denoise_size and median_denoise_size > 0:
            pil_final = pil_final.filter(ImageFilter.MedianFilter(size=max(1, median_denoise_size)))
    except Exception:
        pil_final = Image.fromarray(final)
    out_buf = BytesIO()
    pil_final.convert("RGB").save(out_buf, format="PNG")
    return out_buf.getvalue(), current_peak, scale

# -----------------------------
# Column-matching helpers & RA/DEC inference
# -----------------------------
def pick_first_existing(candidates, cols):
    cols_set = set(cols)
    for c in candidates:
        if c in cols_set:
            return c
    lower_map = {d.lower(): d for d in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def infer_ra_dec_from_values(df):
    cols = df.columns.tolist()
    numeric = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    best_ra = (None, 0.0, False)
    best_dec = (None, 0.0)
    n = len(df)
    if n == 0:
        return None, None, False
    for c in numeric:
        ser = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(ser) < max(10, int(0.01 * max(1, n))):
            continue
        total = len(ser)
        frac_deg360 = ((ser >= 0.0) & (ser <= 360.0)).sum() / total
        frac_hours = ((ser >= 0.0) & (ser <= 24.0)).sum() / total
        frac_dec = ((ser >= -90.0) & (ser <= 90.0)).sum() / total
        if frac_deg360 > 0.90 or frac_hours > 0.90:
            ra_score = max(frac_deg360, frac_hours)
            is_hours = frac_hours > frac_deg360
            if ra_score > best_ra[1]:
                best_ra = (c, ra_score, is_hours)
        if frac_dec > 0.90:
            if frac_dec > best_dec[1]:
                best_dec = (c, frac_dec)
    ra_col = best_ra[0] if best_ra[1] > 0.90 else None
    ra_is_hours = best_ra[2] if ra_col else False
    dec_col = best_dec[0] if best_dec[1] > 0.90 else None
    return ra_col, dec_col, ra_is_hours

# -----------------------------
# Vizier query + normalization
# -----------------------------

# -----------------------------
# Vizier query + normalization
# -----------------------------
# Note: this chunk assumes the rest of your project defines helpers used below
# (e.g. init_requests_session, get_ps1_cutout_simple, get_ps1_cutout_from_filenames,
# get_skyview_cutout, process_strong_background, normalize_star_colors,
# is_effectively_grayscale, pick_first_existing, infer_ra_dec_from_values, etc.)
# If any of those are missing, you'll need to include them elsewhere in your codebase.

import os
import math
import re
import json
import shutil
import zipfile
import threading
import concurrent.futures
import argparse
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Try to use external plugin if available, otherwise use fallback classifier.
try:
    from stellar_stage_classifier import classify_table as external_classify_table
    _HAS_EXTERNAL_CLASSIFIER = True
    print("Using external stellar_stage_classifier.classify_table")
except Exception:
    external_classify_table = None
    _HAS_EXTERNAL_CLASSIFIER = False
    print("External stellar_stage_classifier not found — using fallback classifier")


# -----------------------------
# Vizier query + normalization helpers
# -----------------------------
def query_single_vizier_catalog(catalog, nrows=10000):
    try:
        Vizier.ROW_LIMIT = min(max(nrows, 10), 1000000)
        v = Vizier(columns=['*'], row_limit=Vizier.ROW_LIMIT)
        tables = v.get_catalogs(catalog)
        if not tables or len(tables) == 0:
            print(f"Vizier: no tables returned for catalog '{catalog}'")
            return None
        table = tables[0].to_pandas()
        print(f"Vizier: fetched {len(table)} rows from catalog {catalog}")
        return table
    except Exception as e:
        print(f"Vizier query failed for {catalog}: {e}")
        return None

def normalize_table_for_pipeline(df, catalog_name=None):
    cols = df.columns.tolist()
    sid_candidates = ['source_id', 'Source', 'ID', 'HIP', 'TYC', 'ID_MAIN', 'MainID', 'ID']
    g_candidates = ['phot_g_mean_mag', 'Gmag', 'G', 'Vmag', 'V', 'VTmag', 'Vmag_main', 'Vmag']
    bprp_candidates = ['bp_rp', 'BP-RP', 'B-RP', 'bprp', 'BP_RP']
    bv_candidates = ['B-V', 'BV', 'BminusV', 'B_V', 'B_Vmag']
    plx_candidates = ['Plx', 'parallax', 'plx', 'Plx_mas']
    teff_candidates = ['Teff', 'teff', 'Teff_K']
    sptype_candidates = ['SpType', 'SpType_Vizier', 'sptype', 'SpT', 'SpectralType']
    ra_candidates = ['RA_ICRS','RAJ2000','RAJ2000d','RAJ2000deg','RA','RA_deg','RAdeg','RAJ2000.0','raj2000','RAd']
    dec_candidates = ['DE_ICRS','DEJ2000','DEJ2000d','DEJ2000deg','DE','DE_deg','DEdeg','dej2000','DEd','dedeg']

    sid_col = pick_first_existing(sid_candidates, cols)
    g_col = pick_first_existing(g_candidates, cols)
    bprp_col = pick_first_existing(bprp_candidates, cols)
    bv_col = pick_first_existing(bv_candidates, cols)
    plx_col = pick_first_existing(plx_candidates, cols)
    teff_col = pick_first_existing(teff_candidates, cols)
    sptype_col = pick_first_existing(sptype_candidates, cols)
    ra_col = pick_first_existing(ra_candidates, cols)
    dec_col = pick_first_existing(dec_candidates, cols)

    out = pd.DataFrame()
    out['source_id'] = df[sid_col].astype(str) if sid_col else df.index.astype(str)
    out['phot_g_mean_mag'] = pd.to_numeric(df[g_col], errors='coerce') if g_col else np.nan
    if bprp_col:
        out['bp_rp'] = pd.to_numeric(df[bprp_col], errors='coerce')
    elif bv_col:
        out['bp_rp'] = pd.to_numeric(df[bv_col], errors='coerce') * 1.1
    else:
        out['bp_rp'] = np.nan
    out['parallax'] = pd.to_numeric(df[plx_col], errors='coerce') if plx_col else np.nan
    out['teff_gspphot'] = pd.to_numeric(df[teff_col], errors='coerce') if teff_col else np.nan
    out['sptype'] = df[sptype_col].astype(str) if sptype_col else ""
    if ra_col:
        out['ra_deg'] = pd.to_numeric(df[ra_col], errors='coerce')
    else:
        inf_ra, inf_dec, inf_ra_is_hours = infer_ra_dec_from_values(df)
        if inf_ra:
            out['ra_deg'] = pd.to_numeric(df[inf_ra], errors='coerce')
            if inf_ra_is_hours:
                out['ra_deg'] = out['ra_deg'] * 15.0
        else:
            out['ra_deg'] = np.nan
    if dec_col:
        out['dec_deg'] = pd.to_numeric(df[dec_col], errors='coerce')
    else:
        if 'inf_dec' in locals() and inf_dec:
            out['dec_deg'] = pd.to_numeric(df[inf_dec], errors='coerce')
        else:
            out['dec_deg'] = np.nan
    out['catalog'] = catalog_name if catalog_name else ""
    return out

def query_multiple_vizier_catalogs(catalog_list, nstars, per_catalog_rows=None, max_total_rows=1000000):
    if not catalog_list:
        raise ValueError("No catalogs provided")
    nc = len(catalog_list)
    if per_catalog_rows is None:
        target_total = min(max(nstars * 2, 200), max_total_rows)
        per_catalog_rows = max(200, int(math.ceil(target_total / nc)))
    frames = []
    for cat in catalog_list:
        print(f"\n--- Querying catalog {cat} (row limit {per_catalog_rows}) ---")
        tbl = query_single_vizier_catalog(cat, nrows=per_catalog_rows)
        if tbl is None:
            print(f"Warning: catalog {cat} returned no data or failed; skipping.")
            continue
        try:
            norm = normalize_table_for_pipeline(tbl, catalog_name=cat)
            frames.append(norm)
        except Exception as e:
            print(f"Warning: failed to normalize catalog {cat}: {e}")
            continue
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined raw rows from all catalogs: {len(combined)}")
    if 'source_id' in combined.columns:
        combined['source_id_clean'] = combined['source_id'].replace('', np.nan)
        combined = combined.drop_duplicates(subset=['source_id_clean'], keep='first')
        combined = combined.drop(columns=['source_id_clean'])
    def build_key(row):
        sid = row.get('source_id')
        if sid and str(sid).strip():
            return str(sid)
        cat = row.get('catalog', '')
        mag = row.get('phot_g_mean_mag')
        plx = row.get('parallax')
        mag_r = round(mag, 3) if (mag is not None and not pd.isna(mag)) else 'nm'
        plx_r = round(plx, 3) if (plx is not None and not pd.isna(plx)) else 'np'
        return f"{cat}|{mag_r}|{plx_r}"
    combined['dedupe_key'] = combined.apply(build_key, axis=1)
    combined = combined.drop_duplicates(subset=['dedupe_key'], keep='first')
    combined = combined.drop(columns=['dedupe_key'])
    print(f"After de-duplication: {len(combined)}")
    return combined


# -----------------------------
# Stellar conversions and spectral parsing
# -----------------------------
CLASS_TEMP_RANGES = {
    "O": (50000.0, 30000.0),
    "B": (30000.0, 10000.0),
    "A": (10000.0, 7500.0),
    "F": (7500.0, 6000.0),
    "G": (6000.0, 5200.0),
    "K": (5200.0, 3700.0),
    "M": (3700.0, 2400.0)
}

def bv_to_teff_ballesteros(bv):
    try:
        x = bv
        T = 4600 * (1.0 / (0.92 * x + 1.7) + 1.0 / (0.92 * x + 0.62))
        return float(T)
    except Exception:
        return None

def parse_letter_and_subclass_from_sptype(sptype_str):
    if not sptype_str or not isinstance(sptype_str, str):
        return None, None
    s = sptype_str.strip().upper()
    m = re.match(r'^\s*([OBAFGKM])\s*([0-9](?:\.[0-9])?)?', s)
    if m:
        letter = m.group(1)
        sub = m.group(2)
        if sub is None:
            return letter, None
        try:
            subf = float(sub)
            subi = int(round(subf))
            subi = max(0, min(9, subi))
            return letter, subi
        except Exception:
            return letter, None
    return None, None

def infer_subclass_within_letter(teff, letter):
    if letter is None:
        return None
    if teff is None or (isinstance(teff, float) and np.isnan(teff)):
        return None
    if letter not in CLASS_TEMP_RANGES:
        return None
    thigh, tlow = CLASS_TEMP_RANGES[letter]
    denom = (thigh - tlow) if (thigh - tlow) != 0 else 1.0
    frac = (thigh - float(teff)) / denom
    sub = int(round(max(0, min(9, frac * 9.0))))
    return sub

def derive_spectral_letter_and_subclass(sptype_str, teff=None, bv=None):
    letter, sub = parse_letter_and_subclass_from_sptype(sptype_str)
    source = "sptype"
    if letter:
        if sub is not None:
            return letter, sub, f"{letter}{sub}", source
        if teff is not None and not (isinstance(teff, float) and np.isnan(teff)):
            sub_inf = infer_subclass_within_letter(teff, letter)
            if sub_inf is not None:
                return letter, sub_inf, f"{letter}{sub_inf}", "teff_inferred"
        if bv is not None and not (isinstance(bv, float) and np.isnan(bv)):
            try:
                t_est = bv_to_teff_ballesteros(float(bv))
                if t_est:
                    sub_inf = infer_subclass_within_letter(t_est, letter)
                    if sub_inf is not None:
                        return letter, sub_inf, f"{letter}{sub_inf}", "bv_inferred"
            except Exception:
                pass
        return letter, None, letter, source
    if teff is not None and not (isinstance(teff, float) and np.isnan(teff)):
        T = float(teff)
        for L, (high, low) in CLASS_TEMP_RANGES.items():
            if low - 1 <= T <= high + 1:
                denom = (high - low) if (high - low) != 0 else 1.0
                frac = (high - T) / denom
                sub = int(round(max(0, min(9, frac * 9.0))))
                return L, sub, f"{L}{sub}", "teff_inferred"
    if bv is not None and not (isinstance(bv, float) and np.isnan(bv)):
        try:
            t_est = bv_to_teff_ballesteros(float(bv))
            if t_est:
                T = t_est
                for L, (high, low) in CLASS_TEMP_RANGES.items():
                    if low - 1 <= T <= high + 1:
                        denom = (high - low) if (high - low) != 0 else 1.0
                        frac = (high - T) / denom
                        sub = int(round(max(0, min(9, frac * 9.0))))
                        return L, sub, f"{L}{sub}", "bv_inferred"
        except Exception:
            pass
    return None, None, "", ""


# -----------------------------
# Assemble final dataset (spectral metadata)  -- INTEGRATED with stellar_stage
# -----------------------------
def assemble_final_dataset(df_combined, nstars):
    rows = []
    count = 0
    total = len(df_combined)
    for _, row in tqdm(df_combined.iterrows(), total=total, desc="Assembling spectral-derived metadata"):
        if count >= nstars:
            break
        try:
            source_id = row.get('source_id')
            gmag = row.get('phot_g_mean_mag')
            bp_rp = row.get('bp_rp')
            parallax = row.get('parallax')
            teff = row.get('teff_gspphot')
            sptype = row.get('sptype') if 'sptype' in row else None
            ra_deg = row.get('ra_deg') if 'ra_deg' in row else np.nan
            dec_deg = row.get('dec_deg') if 'dec_deg' in row else np.nan

            bv = None
            if bp_rp is not None and not (isinstance(bp_rp, float) and np.isnan(bp_rp)):
                x = bp_rp
                try:
                    bv = float(0.010 + 0.789 * x - 0.032 * (x ** 2))
                except Exception:
                    bv = None

            def abs_mag_from_parallax(gmag_val, bp_rp_val, parallax_mas):
                try:
                    phot_g = gmag_val
                    if phot_g is None or (isinstance(phot_g, float) and np.isnan(phot_g)):
                        return None
                    if bp_rp_val is None or (isinstance(bp_rp_val, float) and np.isnan(bp_rp_val)):
                        V = phot_g
                    else:
                        V = phot_g + 0.0176 + 0.00686 * bp_rp_val + 0.1732 * (bp_rp_val ** 2)
                    if parallax_mas is None or parallax_mas <= 0:
                        return None
                    dpc = 1000.0 / parallax_mas
                    Mv = V - 5 * (math.log10(dpc) - 1)
                    return float(Mv)
                except Exception:
                    return None

            Mv = abs_mag_from_parallax(gmag, bp_rp, parallax)
            L = None
            if Mv is not None:
                try:
                    L = 10 ** ((4.83 - Mv) / 2.5)
                except Exception:
                    L = None

            teff_final = None
            if teff is not None and not (isinstance(teff, float) and np.isnan(teff)):
                teff_final = float(teff)
            else:
                if bv is not None:
                    teff_final = bv_to_teff_ballesteros(bv)

            R = None
            if L is not None and teff_final is not None:
                try:
                    R = math.sqrt(L) / ((teff_final / 5778.0) ** 2)
                except Exception:
                    R = None

            mass = None
            if L is not None and L > 0:
                try:
                    m = 1.0
                    for _ in range(10):
                        if m < 0.43:
                            alpha = 2.3
                        elif m < 2.0:
                            alpha = 4.0
                        else:
                            alpha = 3.5
                        m = L ** (1.0 / alpha)
                    mass = float(m)
                except Exception:
                    mass = None

            letter, subclass, full_spec, sub_source = derive_spectral_letter_and_subclass(sptype, teff=teff_final, bv=bv)
            spectral_type_str = full_spec if full_spec else (sptype if sptype else "")

            rowdict = {
                "source_id": str(source_id),
                "spectral_type": spectral_type_str,
                "spectral_type_letter": letter if letter else "",
                "spectral_subclass": int(subclass) if subclass is not None else "",
                "subclass_source": sub_source if sub_source else "",
                "category": "",
                "mass": round(mass, 5) if mass else "",
                "luminosity": round(L, 6) if L else "",
                "radius": round(R, 6) if R else "",
                "teff": int(round(teff_final)) if teff_final else "",
                "bv": round(bv, 4) if bv else "",
                "abs_mag": round(Mv, 5) if Mv else "",
                "rgb": "",
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "phot_g_mean_mag": float(gmag) if (gmag is not None and not (isinstance(gmag,float) and np.isnan(gmag))) else ""
            }

            for k, v in row.items():
                try:
                    rowdict[f"input__{k}"] = v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else ""
                except Exception:
                    rowdict[f"input__{k}"] = str(v)

            rows.append(rowdict)
            count += 1
        except Exception:
            continue

    df_out = pd.DataFrame(rows)
    cols = ["source_id", "spectral_type", "spectral_type_letter", "spectral_subclass", "subclass_source",
            "mass", "luminosity", "radius", "teff", "bv", "abs_mag", "rgb", "ra_deg", "dec_deg", "phot_g_mean_mag"]
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ""

    # -----------------------------
    # Stellar stage classification integration
    # -----------------------------
    def classify_simple_row(r):
        """
        Compact fallback classifier returns (stage, score_dict)
        Uses radius, abs_mag, teff, luminosity, and spectral_type hints.
        """
        scores = {
            'white_dwarf': 0.0,
            'main_sequence': 0.0,
            'subgiant': 0.0,
            'giant': 0.0,
            'supergiant': 0.0,
            'unknown': 0.0
        }
        def to_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        radius = to_float(r.get('radius', np.nan))
        abs_mag = to_float(r.get('abs_mag', np.nan))
        teff_v = to_float(r.get('teff', np.nan))
        lum = to_float(r.get('luminosity', np.nan))
        sptype = str(r.get('spectral_type','')).upper() if r.get('spectral_type') not in (None,"") else ""

        if not np.isnan(abs_mag):
            if abs_mag >= 8.0:
                scores['white_dwarf'] += 3.0
            if abs_mag < 2.0:
                scores['giant'] += 2.0
            if 2.0 <= abs_mag < 3.5:
                scores['subgiant'] += 1.5
            if 3.5 <= abs_mag < 8.0:
                scores['main_sequence'] += 2.0

        if not np.isnan(radius):
            if radius < 0.25:
                scores['white_dwarf'] += 2.5
            elif radius < 1.8:
                scores['main_sequence'] += 1.5
            elif radius < 10:
                scores['giant'] += 2.0
            else:
                scores['supergiant'] += 3.0

        if not np.isnan(lum):
            if lum > 1e5:
                scores['supergiant'] += 3.0
            elif lum > 1000:
                scores['giant'] += 2.0
            elif lum > 10:
                scores['main_sequence'] += 1.0

        if sptype:
            if 'WD' in sptype or sptype.startswith('D') or 'DG' in sptype:
                scores['white_dwarf'] += 3.0
            if any(sptype.startswith(pref) for pref in ['I','II','III','IV']):
                if 'III' in sptype:
                    scores['giant'] += 2.0
                if 'IV' in sptype:
                    scores['subgiant'] += 1.5
                if sptype.startswith('I'):
                    scores['supergiant'] += 3.0
            if any(sptype.startswith(c) for c in ['O','B']) and not np.isnan(lum) and lum > 1000:
                scores['supergiant'] += 1.0

        max_score = max(scores.values())
        if max_score <= 0.5:
            final = 'unknown'
        else:
            order = ['white_dwarf','main_sequence','subgiant','giant','supergiant']
            best = max(scores.items(), key=lambda kv: (kv[1], -order.index(kv[0]) if kv[0] in order else 0))[0]
            final = best
        return final, scores

    try:
        if _HAS_EXTERNAL_CLASSIFIER and external_classify_table is not None:
            try:
                classified = external_classify_table(df_out)
                if isinstance(classified, (pd.DataFrame,)):
                    if 'stellar_stage' in classified.columns:
                        df_out['stellar_stage'] = classified['stellar_stage']
                    if 'stellar_stage_score' in classified.columns:
                        df_out['stellar_stage_score'] = classified['stellar_stage_score']
                    else:
                        # if only stage returned, provide a simple score field
                        if 'stellar_stage' in classified.columns:
                            df_out['stellar_stage_score'] = classified['stellar_stage'].apply(lambda s: str({s: 1.0}))
                else:
                    df_out['stellar_stage'], df_out['stellar_stage_score'] = zip(
                        *df_out.apply(lambda r: classify_simple_row(r), axis=1)
                    )
            except Exception as ee:
                print("Warning: external classifier failed, using fallback. Error:", ee)
                df_out['stellar_stage'], df_out['stellar_stage_score'] = zip(
                    *df_out.apply(lambda r: classify_simple_row(r), axis=1)
                )
        else:
            df_out['stellar_stage'], df_out['stellar_stage_score'] = zip(
                *df_out.apply(lambda r: classify_simple_row(r), axis=1)
            )
    except Exception as e:
        print("Warning: stellar stage classification failed:", e)
        df_out['stellar_stage'] = ""
        df_out['stellar_stage_score'] = [str({}) for _ in range(len(df_out))]

    return df_out


# -----------------------------
# Image selection / recenter helpers
# -----------------------------
def central_and_annulus_stats(img_bytes, size_px=256, annulus_r_in=8, annulus_r_out=28, central_box=5):
    eps = 1e-6
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return 0.0, 0.0, 0.0
    if im.size != (size_px, size_px):
        im = im.resize((size_px, size_px), resample=Image.LANCZOS)
    arr = np.array(im).astype(float)
    h, w = arr.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    half = central_box // 2
    y0 = int(round(cy)) - half; y1 = y0 + 2 * half + 1
    x0 = int(round(cx)) - half; x1 = x0 + 2 * half + 1
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    central_patch = arr[y0:y1, x0:x1, :]
    central_peak = float(central_patch.max()) if central_patch.size > 0 else 0.0
    mask_ann = (r >= annulus_r_in) & (r <= annulus_r_out)
    if mask_ann.sum() == 0:
        ann_median = float(np.median(arr.mean(axis=2))) if arr.size > 0 else 0.0
    else:
        ann_median = float(np.median(arr.mean(axis=2)[mask_ann]))
    peak_to_bg = central_peak / (ann_median + eps)
    return central_peak, ann_median, peak_to_bg

def detect_star_presence(img_bytes, size_px=256, annulus_r_in=8, annulus_r_out=28,
                         threshold_abs=20.0, fraction_of_contrast=0.25, min_blob_area=50, max_center_offset=2):
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return False
    if im.size != (size_px, size_px):
        im = im.resize((size_px, size_px), resample=Image.LANCZOS)
    arr = np.array(im).astype(float)
    intensity = arr.mean(axis=2)
    h, w = intensity.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask_ann = (r >= annulus_r_in) & (r <= annulus_r_out)
    if mask_ann.sum() == 0:
        ann_median = float(np.median(intensity)) if intensity.size > 0 else 0.0
    else:
        ann_median = float(np.median(intensity[mask_ann]))
    half = 2
    y0 = int(round(cy)) - half; y1 = y0 + 2 * half + 1
    x0 = int(round(cx)) - half; x1 = x0 + 2 * half + 1
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    central_peak = float(intensity[y0:y1, x0:x1].max()) if intensity[y0:y1, x0:x1].size > 0 else float(intensity.max())
    dyn_thresh = ann_median + max(threshold_abs, fraction_of_contrast * max(0.0, central_peak - ann_median))
    mask = intensity > dyn_thresh
    if mask.sum() == 0:
        return False
    cy_i, cx_i = int(round(cy)), int(round(cx))
    if not mask[cy_i, cx_i]:
        neigh = mask[max(0, cy_i - 1):min(h, cy_i + 2), max(0, cx_i - 1):min(w, cx_i + 2)]
        if neigh.sum() == 0:
            return False
        coords = np.argwhere(neigh)
        rel = coords[0]
        seed_y = max(0, cy_i - 1) + int(rel[0])
        seed_x = max(0, cx_i - 1) + int(rel[1])
    else:
        seed_y, seed_x = cy_i, cx_i
    hmask = mask
    visited = np.zeros_like(hmask, dtype=bool)
    stack = [(seed_y, seed_x)]
    comp_coords = []
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if not hmask[y, x] or visited[y, x]:
            continue
        visited[y, x] = True
        comp_coords.append((y, x))
        stack.append((y - 1, x)); stack.append((y + 1, x))
        stack.append((y, x - 1)); stack.append((y, x + 1))
    if len(comp_coords) < min_blob_area:
        return False
    coords = np.array(comp_coords)
    ys = coords[:, 0]; xs = coords[:, 1]
    centroid_y = ys.mean(); centroid_x = xs.mean()
    max_intensity = intensity[ys, xs].max()
    dist_center = np.sqrt((centroid_x - cx) ** 2 + (centroid_y - cy) ** 2)
    if dist_center > max_center_offset:
        return False
    if max_intensity <= ann_median + 1.0:
        return False
    return True

def recenter_by_large_cutout(ra, dec, peak_search_size, size_px, try_fetch_func):
    try:
        res = try_fetch_func(ra, dec, peak_search_size)
        if not res:
            return None, None
        big = res[0] if isinstance(res, (list, tuple)) else res
        if not big:
            return None, None
        im_big = Image.open(BytesIO(big)).convert("RGB")
    except Exception:
        try:
            big = get_ps1_cutout_simple(ra, dec, size_px=peak_search_size)
            if not big:
                return None, None
            im_big = Image.open(BytesIO(big)).convert("RGB")
        except Exception:
            return None, None
    bx, by = im_big.size
    arr = np.array(im_big).astype(float)
    intensity = arr.mean(axis=2)
    py, px = np.unravel_index(np.nanargmax(intensity), intensity.shape)
    left = int(max(0, px - size_px // 2))
    upper = int(max(0, py - size_px // 2))
    right = left + size_px
    lower = upper + size_px
    if right > bx:
        right = bx; left = max(0, bx - size_px)
    if lower > by:
        lower = by; upper = max(0, by - size_px)
    try:
        crop = im_big.crop((left, upper, right, lower))
        buf = BytesIO(); crop.save(buf, format="PNG")
        return buf.getvalue(), f"large_crop_at_{py}_{px}"
    except Exception:
        return None, None

def recenter_by_roll(img_bytes, size_px):
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGB")
        if im.size != (size_px, size_px):
            im = im.resize((size_px, size_px), resample=Image.LANCZOS)
        arr = np.array(im).astype(float)
        intensity = arr.mean(axis=2)
        h, w = intensity.shape
        py, px = np.unravel_index(np.nanargmax(intensity), intensity.shape)
        cy, cx = h // 2, w // 2
        dy = cy - py
        dx = cx - px
        rolled = np.zeros_like(arr)
        for ch in range(3):
            rolled[:, :, ch] = np.roll(np.roll(arr[:, :, ch], dy, axis=0), dx, axis=1)
        rolled = np.clip(rolled, 0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(rolled).convert("RGB").save(buf, format="PNG")
        return buf.getvalue(), f"rolled_{dy}_{dx}"
    except Exception:
        return None, None


# -----------------------------
# Canonicalize helper
# -----------------------------
def _canonicalize_value(v):
    if v is None:
        return ""
    if isinstance(v, (np.floating, float)):
        if np.isnan(v):
            return ""
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if pd.isna(v):
        return ""
    try:
        return v.item() if hasattr(v, "item") else v
    except Exception:
        return str(v)


# -----------------------------
# Build images & pack (main routine) — optimized with ThreadPoolExecutor
# -----------------------------
def build_images_and_pack(df_meta, nstars=1000, out_dir_raw="star_images_raw",
                          out_dir_proc="star_images_proc", out_csv="star_images_catalog.csv",
                          out_zip="star_images_bundle.zip", size_px=256,
                          min_peak=75.0, min_peak_bg_ratio=5, min_contrast=30.0,
                          try_larger_on_empty=True,
                          peak_search_size_multiplier=5, verbose=True,
                          fetch_workers=8, batch_size=16, timeout=15,
                          use_gpu=False):
    os.makedirs(out_dir_raw, exist_ok=True)
    os.makedirs(out_dir_proc, exist_ok=True)
    sess = init_requests_session(timeout=timeout)

    candidates = df_meta.copy().reset_index(drop=True)
    candidates = candidates[candidates['ra_deg'].notna() & candidates['dec_deg'].notna()].reset_index(drop=True)
    if len(candidates) == 0:
        print("No candidates with coordinates to fetch images for.")
        return None

    results = []
    images_list = []
    saved = 0
    saved_lock = threading.Lock()

    base_target_peak = 180.0
    factor_clip_min = 0.2
    factor_clip_max = 5.0

    fetch_cache = {}
    cache_lock = threading.Lock()

    def try_fetch_color_first_local(ra, dec, desired_size):
        key_try = (float(ra), float(dec), int(desired_size))
        with cache_lock:
            if key_try in fetch_cache:
                return fetch_cache[key_try]
        try:
            b = get_ps1_cutout_simple(ra, dec, size_px=desired_size, timeout=timeout)
            if b:
                out = (b, "PS1_simple")
                with cache_lock:
                    fetch_cache[key_try] = out
                return out
        except Exception:
            pass
        try:
            b = get_ps1_cutout_from_filenames(ra, dec, size_px=desired_size, filters="gri", timeout=timeout)
            if b:
                out = (b, "PS1_filenames")
                with cache_lock:
                    fetch_cache[key_try] = out
                return out
        except Exception:
            pass
        try:
            b = get_skyview_cutout(ra, dec, size_px=desired_size, survey="DSS2 Red", timeout=timeout)
            if b:
                out = (b, "DSS2")
                with cache_lock:
                    fetch_cache[key_try] = out
                return out
        except Exception:
            pass
        try:
            b = get_skyview_cutout(ra, dec, size_px=desired_size, survey="2MASS-J", timeout=timeout)
            if b:
                out = (b, "2MASS")
                with cache_lock:
                    fetch_cache[key_try] = out
                return out
        except Exception:
            pass
        with cache_lock:
            fetch_cache[key_try] = (None, None)
        return None, None

    skip_counters = {"no_image":0, "too_faint":0, "grayscale":0, "failed_proc":0, "exception":0, "failed_proc_save":0}

    # worker returns success dict with proc_bytes + raw_bytes instead of saving to disk
    def process_candidate(idx, row):
        try:
            ra = float(row['ra_deg']); dec = float(row['dec_deg'])
        except Exception:
            return None
        base_meta = {}
        for k, v in row.items():
            base_meta[str(k)] = _canonicalize_value(v)
        img_bytes, survey = try_fetch_color_first_local(ra, dec, size_px)
        if not img_bytes:
            return {"skip": "no_image"}
        if is_effectively_grayscale(img_bytes):
            alt_bytes = get_ps1_cutout_from_filenames(ra, dec, size_px=size_px)
            if alt_bytes and not is_effectively_grayscale(alt_bytes):
                img_bytes = alt_bytes; survey = "PS1_filenames_fix"
            else:
                return {"skip": "grayscale"}
        central_peak, ann_median, peak_to_bg = central_and_annulus_stats(
            img_bytes, size_px=size_px,
            annulus_r_in=max(6, int(size_px*0.12)),
            annulus_r_out=max(20, int(size_px*0.45)),
            central_box=5
        )
        contrast_abs = central_peak - ann_median
        passes_ratio = (peak_to_bg >= min_peak_bg_ratio and central_peak >= min_peak)
        passes_contrast = (contrast_abs >= min_contrast and central_peak >= min_peak)
        accepted = False
        if passes_ratio or passes_contrast:
            accepted = True
        else:
            big_bytes, big_survey = try_fetch_color_first_local(ra, dec, size_px*2)
            cropped = None
            if big_bytes:
                try:
                    im_big = Image.open(BytesIO(big_bytes)).convert("RGB")
                    bx, by = im_big.size; cx, cy = bx//2, by//2
                    left = max(0, cx - size_px//2); upper = max(0, cy - size_px//2)
                    right = left + size_px; lower = upper + size_px
                    crop = im_big.crop((left, upper, right, lower))
                    buf = BytesIO(); crop.save(buf, format="PNG")
                    cropped = buf.getvalue()
                    central_peak2, ann_median2, peak_to_bg2 = central_and_annulus_stats(
                        cropped, size_px=size_px,
                        annulus_r_in=max(6, int(size_px*0.12)),
                        annulus_r_out=max(20, int(size_px*0.45)),
                        central_box=5
                    )
                    contrast_abs2 = central_peak2 - ann_median2
                    passes_ratio2 = (peak_to_bg2 >= min_peak_bg_ratio and central_peak2 >= min_peak)
                    passes_contrast2 = (contrast_abs2 >= min_contrast and central_peak2 >= min_peak)
                    if passes_ratio2 or passes_contrast2:
                        img_bytes = cropped; survey = big_survey + "_crop"
                        central_peak, ann_median, peak_to_bg = central_peak2, ann_median2, peak_to_bg2
                        accepted = True
                except Exception:
                    pass
            if not accepted:
                blob_ok = detect_star_presence(img_bytes, size_px=size_px,
                                               annulus_r_in=max(6, int(size_px*0.12)),
                                               annulus_r_out=max(20, int(size_px*0.45)),
                                               threshold_abs=10.0, fraction_of_contrast=0.20,
                                               min_blob_area=6, max_center_offset=4)
                if not blob_ok and cropped is not None:
                    try:
                        blob_ok2 = detect_star_presence(cropped, size_px=size_px,
                                                        annulus_r_in=max(6, int(size_px*0.12)),
                                                        annulus_r_out=max(20, int(size_px*0.45)),
                                                        threshold_abs=10.0, fraction_of_contrast=0.20,
                                                        min_blob_area=6, max_center_offset=4)
                        blob_ok = blob_ok or blob_ok2
                    except Exception:
                        pass
                if blob_ok:
                    accepted = True
        if not accepted:
            return {"skip": "too_faint"}
        centered_bytes = img_bytes
        recentered = False; recenter_method = ""
        try:
            large_crop, method = recenter_by_large_cutout(ra, dec, peak_search_size=size_px * peak_search_size_multiplier,
                                                          size_px=size_px, try_fetch_func=lambda ra_,dec_,sz: try_fetch_color_first_local(ra_,dec_,sz))
            if large_crop:
                centered_bytes = large_crop; recentered = True; recenter_method = method
        except Exception:
            pass
        if not recentered:
            rolled, method = recenter_by_roll(img_bytes, size_px=size_px)
            if rolled:
                centered_bytes = rolled; recentered = True; recenter_method = method
        apparent_mag = None
        try:
            am = base_meta.get("phot_g_mean_mag", None)
            if am is not None and am != "":
                apparent_mag = float(am)
        except Exception:
            apparent_mag = None
        abs_mag_val = None
        try:
            amv = base_meta.get("abs_mag", None)
            if amv is not None and amv != "":
                abs_mag_val = float(amv)
            else:
                plx = base_meta.get("parallax", None)
                bp_rp = base_meta.get("bp_rp", None) if "bp_rp" in base_meta else None
                if apparent_mag is not None and plx not in (None, "", 0) and float(plx) > 0:
                    phot_g = apparent_mag
                    V = phot_g if (bp_rp in (None, "")) else (phot_g + 0.0176 + 0.00686 * float(bp_rp) + 0.1732 * (float(bp_rp) ** 2))
                    abs_mag_val = V - 5 * (math.log10(1000.0/float(plx)) - 1)
        except Exception:
            abs_mag_val = None
        target_peak_for_star = base_target_peak
        try:
            if abs_mag_val is not None:
                factor = 10.0 ** (-0.4 * (abs_mag_val - 4.83))
                factor = max(factor_clip_min, min(factor_clip_max, factor))
                target_peak_for_star = base_target_peak * factor
            else:
                if apparent_mag is not None:
                    ref_app = 10.0
                    factor_app = 10.0 ** (-0.4 * (apparent_mag - ref_app))
                    factor_app = max(0.5, min(2.0, factor_app))
                    target_peak_for_star = base_target_peak * factor_app
                else:
                    target_peak_for_star = base_target_peak
        except Exception:
            target_peak_for_star = base_target_peak
        try:
            proc_bytes, peak_before, scale_used = process_strong_background(
                centered_bytes, size_px=size_px,
                annulus_r_in=max(10, int(size_px*0.15)),
                annulus_r_out=max(20, int(size_px*0.45)),
                lowfreq_blur=10, target_peak=target_peak_for_star,
                peak_clamp=(0.4, 6.0), contrast_percentiles=(1,99),
                unsharp_radius=1.0, median_denoise_size=1,
                use_gpu=use_gpu and _HAS_TORCH and torch.cuda.is_available(),
                gpu_device=(torch.device("cuda") if (_HAS_TORCH and torch.cuda.is_available()) else None)
            )
        except Exception:
            proc_bytes = None; peak_before = None; scale_used = None
        if proc_bytes is None:
            return {"skip": "failed_proc"}
        try:
            im_proc = Image.open(BytesIO(proc_bytes)).convert("RGB")
            if im_proc.size != (size_px, size_px):
                im_proc = im_proc.resize((size_px, size_px), resample=Image.LANCZOS)
            arr_proc = np.array(im_proc).astype(np.uint8)
        except Exception:
            return {"skip": "failed_proc_arr"}
        teff_for_norm = base_meta.get("teff", None)
        try:
            teff_val = None
            if teff_for_norm not in (None, "", np.nan):
                try:
                    teff_val = float(teff_for_norm)
                except Exception:
                    teff_val = None
            arr_proc_norm = normalize_star_colors(arr_proc, teff=teff_val)
            buf2 = BytesIO()
            Image.fromarray(arr_proc_norm).convert("RGB").save(buf2, format="PNG")
            proc_bytes_norm = buf2.getvalue()
        except Exception:
            proc_bytes_norm = proc_bytes
        avg_rgb = None
        try:
            arr_tmp = np.array(Image.open(BytesIO(proc_bytes_norm)).convert("RGB")).astype(np.uint8)
            avg_rgb = arr_tmp.mean(axis=(0,1)).tolist()
            luma = (0.2126*arr_tmp[:,:,0] + 0.7152*arr_tmp[:,:,1] + 0.0722*arr_tmp[:,:,2]).mean()
        except Exception:
            avg_rgb = [0.0,0.0,0.0]; luma = 0.0
        img_fields = {
            "proc_filename": "",
            "apparent_mag": (apparent_mag if apparent_mag is not None else ""),
            "abs_mag_standardized": (round(abs_mag_val, 5) if abs_mag_val is not None else ""),
            "target_peak_for_star": float(target_peak_for_star),
            "parallax_mas": base_meta.get("parallax", ""),
            "distance_pc": (1000.0/float(base_meta["parallax"]) if (base_meta.get("parallax") not in (None,"") and float(base_meta.get("parallax"))>0) else ""),
            "ra_deg": ra, "dec_deg": dec, "survey_used": survey,
            "recentered": recentered, "recenter_method": recenter_method,
            "central_peak": central_peak, "annulus_median": ann_median,
            "peak_to_bg": peak_to_bg, "proc_peak_before": peak_before, "scale_used": scale_used,
            "avg_r": float(avg_rgb[0]), "avg_g": float(avg_rgb[1]), "avg_b": float(avg_rgb[2]), "luma": float(luma)
        }
        merged = dict(base_meta)
        for k, v in img_fields.items():
            merged[str(k)] = _canonicalize_value(v) if not isinstance(v, (str, int, float, bool)) else v
        return {"success": {"meta": merged, "raw_bytes": img_bytes, "proc_bytes": proc_bytes_norm}}

    total_candidates = len(candidates)
    indices = list(range(total_candidates))

    with concurrent.futures.ThreadPoolExecutor(max_workers=fetch_workers) as executor:
        futures = {}
        submitted = 0
        completed_count = 0
        while submitted < min(batch_size, total_candidates):
            i = indices[submitted]
            fut = executor.submit(process_candidate, i, candidates.iloc[i])
            futures[fut] = i
            submitted += 1

        pbar = tqdm(total=min(nstars, total_candidates), desc="Saving images", unit="img")
        while futures and saved < nstars:
            done, _ = concurrent.futures.wait(list(futures.keys()), return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                idx = futures.pop(fut)
                completed_count += 1
                try:
                    res = fut.result()
                except Exception:
                    res = {"skip": "exception"}
                if res is None:
                    pass
                elif "skip" in res:
                    sk = res.get("skip", "unknown")
                    skip_counters.setdefault(sk, 0)
                    skip_counters[sk] += 1
                elif "success" in res:
                    payload = res["success"]
                    merged = payload.get("meta", {})
                    raw_bytes = payload.get("raw_bytes")
                    proc_bytes = payload.get("proc_bytes")
                    with saved_lock:
                        cur_saved = saved
                        new_raw = os.path.join(out_dir_raw, f"star_raw_{cur_saved:05d}.png")
                        new_proc = os.path.join(out_dir_proc, f"star_{cur_saved:05d}.png")
                        try:
                            if raw_bytes:
                                try:
                                    with open(new_raw, "wb") as fh:
                                        fh.write(raw_bytes)
                                    merged["raw_file"] = new_raw
                                except Exception:
                                    merged["raw_file"] = ""
                            else:
                                merged["raw_file"] = ""
                            if proc_bytes:
                                try:
                                    with open(new_proc, "wb") as fh:
                                        fh.write(proc_bytes)
                                    merged["proc_file"] = new_proc
                                except Exception:
                                    merged["proc_file"] = ""
                            else:
                                merged["proc_file"] = ""
                            merged["proc_filename"] = os.path.basename(merged.get("proc_file", "")) or ""
                            results.append(merged)
                            try:
                                if merged.get("proc_file"):
                                    im_proc = Image.open(merged["proc_file"]).convert("RGB")
                                    if im_proc.size != (size_px, size_px):
                                        im_proc = im_proc.resize((size_px, size_px), resample=Image.LANCZOS)
                                    arr_proc = np.array(im_proc).astype(np.uint8)
                                    images_list.append(arr_proc)
                            except Exception:
                                pass
                            saved += 1
                            pbar.update(1)
                        except Exception:
                            skip_counters.setdefault("failed_proc_save", 0)
                            skip_counters["failed_proc_save"] += 1
                if submitted < total_candidates:
                    i = indices[submitted]
                    fut2 = executor.submit(process_candidate, i, candidates.iloc[i])
                    futures[fut2] = i
                    submitted += 1
        pbar.close()

    if len(results) == 0:
        print("No images saved in this run.")
        return None

    meta_df = pd.DataFrame(results)
    meta_df.to_csv(out_csv, index=False)
    if verbose:
        print(f"Wrote metadata CSV -> {out_csv}")

    dataset_dir = "dataset"
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for _, r in meta_df.iterrows():
        src = r.get("proc_file", "")
        fname = r.get("proc_filename", "")
        if src and fname:
            dst = os.path.join(images_dir, fname)
            try:
                shutil.copyfile(src, dst)
            except Exception:
                pass

    essential_cols = []
    possible_cols = list(meta_df.columns)
    for c in ['proc_filename','spectral_type_letter','spectral_subclass','ra_deg','dec_deg','phot_g_mean_mag','teff','avg_r','avg_g','avg_b','luma']:
        if c in possible_cols:
            essential_cols.append(c)
    if 'proc_filename' not in essential_cols:
        meta_df['proc_filename'] = meta_df.apply(lambda r: os.path.basename(r.get('proc_file','')) if r.get('proc_file') else "", axis=1)
        if 'proc_filename' not in essential_cols:
            essential_cols.insert(0, 'proc_filename')
    meta_min = meta_df[essential_cols].copy()
    dataset_meta_path = os.path.join(dataset_dir, "metadata.csv")
    meta_min.to_csv(dataset_meta_path, index=False)

    try:
        with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(dataset_meta_path, arcname=os.path.join(os.path.basename(dataset_dir), "metadata.csv"))
            for root, _, files in os.walk(images_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    arcname = os.path.join(os.path.basename(dataset_dir), "images", fn)
                    zf.write(full, arcname=arcname)
        archive_path = out_zip
    except Exception:
        archive_path = shutil.make_archive(os.path.splitext(out_zip)[0], 'zip', root_dir=dataset_dir)
        if archive_path != out_zip:
            try:
                shutil.move(archive_path, out_zip)
                archive_path = out_zip
            except Exception:
                pass

    if verbose:
        print(f"Saved dataset archive -> {archive_path}")

    return {"n_saved": len(images_list), "out_zip": archive_path, "out_csv": out_csv, "meta_df": meta_df, "images_shape": np.stack(images_list, axis=0).shape}


# -----------------------------
# Export training arrays for CNN (images + letter/subclass labels)
# -----------------------------
def export_training_arrays(meta_df, images_dir="dataset/images", out_prefix="training_data", size_px=256, classes_order=None, verbose=True):
    if classes_order is None:
        classes_order = ['O','B','A','F','G','K','M']
    rows = []
    for _, r in meta_df.iterrows():
        fname = r.get("proc_filename", "")
        if not fname:
            continue
        fpath = os.path.join(images_dir, fname)
        if os.path.exists(fpath):
            rows.append((fpath, fname, r))
    if len(rows) == 0:
        raise RuntimeError("No processed images found in dataset/images to export. Make sure build_images_and_pack created dataset/images.")

    X_list = []
    y_letter = []
    y_subclass = []
    meta_rows = []
    for fpath, fname, row in tqdm(rows, desc="Loading images for export"):
        try:
            im = Image.open(fpath).convert("RGB")
            if im.size != (size_px, size_px):
                im = im.resize((size_px, size_px), resample=Image.LANCZOS)
            arr = np.array(im).astype(np.uint8)
        except Exception:
            continue
        X_list.append(arr)
        letter = row.get("spectral_type_letter", "")
        subclass_raw = row.get("spectral_subclass", "")
        letter_clean = str(letter).upper().strip() if letter not in (None, "") else ""
        try:
            sub_i = int(subclass_raw) if (subclass_raw not in (None,"")) else -1
        except Exception:
            sub_i = -1
        y_subclass.append(sub_i)
        y_letter.append(letter_clean)
        rowcopy = dict(row)
        rowcopy["proc_filepath"] = fpath
        meta_rows.append(rowcopy)

    present_letters = sorted(set([l for l in y_letter if l]), key=lambda x: ('OBAFGKM'.index(x) if x in 'OBAFGKM' else 100))
    classes = [c for c in classes_order if c in present_letters]
    for l in sorted(set(present_letters) - set(classes)):
        classes.append(l)
    class_map = {c: i for i, c in enumerate(classes)}

    y_letter_idx = []
    for l in y_letter:
        if not l or (l not in class_map):
            y_letter_idx.append(-1)
        else:
            y_letter_idx.append(class_map[l])

    X = np.stack(X_list, axis=0).astype(np.uint8)
    y_letter_arr = np.array(y_letter_idx, dtype=np.int32)
    y_sub_arr = np.array(y_subclass, dtype=np.int32)

    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y_letter.npy", y_letter_arr)
    np.save(f"{out_prefix}_y_subclass.npy", y_sub_arr)
    with open(f"{out_prefix}_classes.json", "w") as fh:
        json.dump({"classes": classes, "map": class_map}, fh, indent=2)
    meta_out = pd.DataFrame(meta_rows)
    meta_out.to_csv(f"{out_prefix}_metadata_for_training.csv", index=False)
    if verbose:
        print(f"Saved {out_prefix}_X.npy ({X.shape}), {out_prefix}_y_letter.npy, {out_prefix}_y_subclass.npy, {out_prefix}_classes.json")
    return {"X_shape": X.shape, "n_samples": X.shape[0], "classes": classes, "meta_csv": f"{out_prefix}_metadata_for_training.csv"}


# -----------------------------
# CLI entry
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build star images + metadata and export training arrays (optimized + GPU optional)")
    parser.add_argument("--nstars", type=int, default=1000, help="Number of stars/images to produce")
    parser.add_argument("--catalogs", type=str, default="I/239/hip_main,I/355/gaiadr3,I/345/gaia2,II/246/out,V/156,II/349/ps1,I/311/hip2,II/328/allwise,V/147/sdss12,V/133/kic", help="Vizier catalog IDs (comma separated)")
    parser.add_argument("--per-catalog", type=int, default=500, help="Rows to fetch per catalog")
    parser.add_argument("--out-meta", type=str, default="stars_dataset.csv", help="Output spectral metadata CSV")
    parser.add_argument("--out-images-zip", type=str, default="star_images_bundle.zip", help="Bundled ZIP with images + CSV")
    parser.add_argument("--out-images-csv", type=str, default="star_images_catalog.csv", help="CSV describing images (detailed)")
    parser.add_argument("--size", type=int, default=256, help="Image size (px)")
    parser.add_argument("--export-npy", action="store_true", help="Export training .npy arrays after building images")
    parser.add_argument("--out-npy-prefix", type=str, default="training_data", help="Prefix for exported .npy files")
    parser.add_argument("--peak-search-multiplier", type=int, default=5, help="Multiplier for large cutout when re-centering")
    parser.add_argument("--fetch-workers", type=int, default=8, help="Number of parallel fetch/worker threads")
    parser.add_argument("--batch-size", type=int, default=32, help="How many candidates to keep submitted concurrently")
    parser.add_argument("--timeout", type=int, default=15, help="Network timeout seconds")
    parser.add_argument("--use-gpu", action="store_true", help="Attempt to use GPU-accelerated array ops (requires torch + CUDA)")
    parser.add_argument("--gpu-benchmark", action="store_true", help="Run a PyTorch CPU vs GPU benchmark (timed forward) and print results")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown args:", unknown)

    if args.gpu_benchmark:
        print("Running GPU benchmark (if torch is installed)...")
        run_gpu_benchmark()

    catalog_list = [c.strip() for c in args.catalogs.split(",") if c.strip()]
    print("Querying catalogs and normalizing...")
    combined = query_multiple_vizier_catalogs(catalog_list, nstars=args.nstars, per_catalog_rows=args.per_catalog)
    if combined is None or len(combined) == 0:
        print("No data from catalogs; exiting.")
        return

    final_meta = assemble_final_dataset(combined, nstars=args.nstars)
    print(f"Final metadata rows prepared: {len(final_meta)}")
    final_meta.to_csv(args.out_meta, index=False)
    print(f"Wrote spectral metadata CSV -> {args.out_meta}")

    image_pack = build_images_and_pack(final_meta, nstars=args.nstars,
                                       out_dir_raw="star_images_raw",
                                       out_dir_proc="star_images_proc",
                                       out_csv=args.out_images_csv,
                                       out_zip=args.out_images_zip,
                                       size_px=args.size,
                                       peak_search_size_multiplier=int(args.peak_search_multiplier),
                                       fetch_workers=int(args.fetch_workers),
                                       batch_size=int(args.batch_size),
                                       timeout=int(args.timeout),
                                       use_gpu=bool(args.use_gpu))
    if image_pack:
        print("Image pack summary:", image_pack)
    else:
        print("Image pack failed or saved zero images.")

    if args.export_npy:
        dataset_meta_path = os.path.join("dataset", "metadata.csv")
        if not os.path.exists(dataset_meta_path):
            print("No dataset/metadata.csv found to export training arrays. Skipping export.")
        else:
            meta_df = pd.read_csv(dataset_meta_path)
            try:
                res = export_training_arrays(meta_df, images_dir="dataset/images", out_prefix=args.out_npy_prefix, size_px=args.size)
                print("Exported training arrays:", res)
            except Exception as e:
                print("Failed to export training arrays:", e)

if __name__ == "__main__":
    main()
