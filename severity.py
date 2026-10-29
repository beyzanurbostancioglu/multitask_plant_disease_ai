"""Klasik görüntü işleme ile hastalıklı alan oranı (lezyon alanı / yaprak alanı).

Yöntem: GrabCut ile yaprak maskesi -> HSV'de 'sağlıklı yeşil' dışındaki pikseller lezyon.
NOT: Bu bir sezgisel (heuristic) yöntemdir; AI Challenger'da piksel düzeyinde maske yoktur.
Esikleri calibrate_severity.py ile veri setindeki Sağlıklı/Genel/Ciddi etiketlerine göre ayarla.
"""
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PARAMS = dict(
    green_h=(33, 90),      # OpenCV H aralığı (0-179) - sağlıklı yeşil
    green_s_min=35, green_v_min=35,
    glare_s_max=30, glare_v_min=210,   # parlama/yansıma: yapraktan ve lezyondan çıkarılır
    max_side=400,
    open_kernel=3,
    min_lesion_px=12,
)
DEFAULT_THRESHOLDS = dict(t_general=2.0, t_serious=15.0)   # % cinsinden; kalibrasyonla değişir


def _leaf_mask(bgr, p):
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    m = int(0.04 * min(h, w))
    rect = (m, m, w - 2 * m, h - 2 * m)
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        leaf = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    except cv2.error:
        leaf = np.ones((h, w), bool)
    frac = leaf.mean()
    if frac < 0.12 or frac > 0.98:     # GrabCut başarısız -> renk tabanlı yedek
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        leaf = (hsv[..., 1] > 35) & (hsv[..., 2] > 30)
    leaf = cv2.morphologyEx(leaf.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((7, 7), np.uint8)).astype(bool)
    return leaf


def estimate_ratio(bgr, params=None):
    """Döndürür: (oran_yüzde, lezyon_maskesi, yaprak_maskesi) - maskeler yeniden boyutlu görüntüye ait."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    h, w = bgr.shape[:2]
    s = p["max_side"] / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    leaf = _leaf_mask(bgr, p)
    glare = (S < p["glare_s_max"]) & (V > p["glare_v_min"])
    leaf &= ~glare
    green = (H >= p["green_h"][0]) & (H <= p["green_h"][1]) & \
            (S >= p["green_s_min"]) & (V >= p["green_v_min"])
    lesion = (leaf & ~green).astype(np.uint8)
    k = np.ones((p["open_kernel"],) * 2, np.uint8)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(lesion)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < p["min_lesion_px"]:
            lesion[lab == i] = 0
    lesion = lesion.astype(bool)
    area = leaf.sum()
    ratio = 100.0 * lesion.sum() / area if area > 0 else 0.0
    return float(ratio), lesion, leaf, bgr


def ratio_to_level(ratio, th=None):
    th = th or DEFAULT_THRESHOLDS
    return 0 if ratio < th["t_general"] else (1 if ratio < th["t_serious"] else 2)


def make_overlay(bgr_small, lesion, leaf):
    out = bgr_small.copy()
    red = np.zeros_like(out); red[lesion] = (0, 0, 255)
    out = np.where(lesion[..., None], cv2.addWeighted(out, 0.45, red, 0.55, 0), out)
    cnts, _ = cv2.findContours(leaf.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 200, 0), 2)
    return out


def load_thresholds(path="severity_thresholds.json"):
    f = Path(path)
    return json.loads(f.read_text()) if f.exists() else DEFAULT_THRESHOLDS


def analyze(bgr, thresholds=None):
    ratio, lesion, leaf, small = estimate_ratio(bgr)
    return dict(ratio=ratio, level=ratio_to_level(ratio, thresholds or load_thresholds()),
                overlay=make_overlay(small, lesion, leaf))
