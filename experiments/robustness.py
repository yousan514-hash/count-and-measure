"""撮り方の条件を1つずつ悪くして、どこで失敗するかを調べる（PDCA の Do / Check）。

元にするのは本物の写真 tests/data/japanese_coins.jpg（硬貨5枚、10円玉を物差し）。
条件ごとに強さを変えた画像を作り、個数と直径の誤差を表にする。

    pixi run python experiments/robustness.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from count_and_measure import measure_image  # noqa: E402
from count_and_measure.cli import _read_image  # noqa: E402
from count_and_measure.measure import MarkerNotFoundError  # noqa: E402

TRUTH = [20.0, 21.0, 22.0, 22.6, 23.5]  # 1円, 50円, 5円, 100円, 10円
REF_MM = 23.5
OUT = ROOT / "experiments" / "out"


# ---- 条件（画像を悪くする関数）。strength は 0（そのまま）〜 大きいほど悪い ----

def tilt(img, deg):
    """カメラを傾けて撮った（台形にゆがむ）。"""
    h, w = img.shape[:2]
    d = np.tan(np.radians(deg)) * h * 0.35
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, 0], [w - d, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)


def dark(img, factor):
    """暗い部屋で撮った（明るさが下がり、ノイズが増える）。"""
    rng = np.random.default_rng(0)
    out = img.astype(np.float32) * (1 - factor)
    out += rng.normal(0, 3 + 20 * factor, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(img, k):
    """ピンぼけ・手ぶれ。"""
    k = int(k) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img


def noise(img, sigma):
    """スマホの暗所ノイズ。"""
    rng = np.random.default_rng(1)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def shadow(img, strength):
    """片側に強い影（窓の光だけで撮った）。"""
    h, w = img.shape[:2]
    grad = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)


def touching(img, px):
    """硬貨を寄せて置いた（右下の1円玉を左に動かして10円玉に近づける）。px は動かす量。"""
    if px == 0:
        return img
    out = img.copy()
    y0, y1, x0, x1 = 640, 1070, 1130, 1540  # 1円玉のまわり
    patch = img[y0:y1, x0:x1].copy()
    bg = np.median(img[20:80, 20:80].reshape(-1, 3), axis=0)
    out[y0:y1, x0:x1] = bg
    nx0 = max(0, x0 - int(px))
    # 背景と同じくらい明るい画素は貼らない（四角い切れ目を作らない）
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    keep = gray < np.median(gray[:20, :20]) - 12
    region = out[y0:y1, nx0:nx0 + patch.shape[1]]
    region[keep[:, :region.shape[1]]] = patch[:, :region.shape[1]][keep[:, :region.shape[1]]]
    return out


def background(img, level):
    """背景の色を変える（level: 白い紙 0 → 1 で灰色の机）。硬貨と背景の明るさの差が小さくなる。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg_mask = (gray > 200).astype(np.float32)
    bg_mask = cv2.GaussianBlur(bg_mask, (15, 15), 0)[..., None]
    target = np.array([150, 150, 150], np.float32)
    mixed = img.astype(np.float32) * (1 - level * bg_mask) + target * level * bg_mask
    return np.clip(mixed, 0, 255).astype(np.uint8)


CONDITIONS = {
    "傾き(度)": (tilt, [0, 10, 20, 30, 40]),
    "暗さ": (dark, [0, 0.3, 0.5, 0.7, 0.85]),
    "ぼけ": (blur, [0, 3, 7, 12, 20]),
    "ノイズ": (noise, [0, 10, 25, 40, 60]),
    "片側の影": (shadow, [0, 0.3, 0.5, 0.7, 0.85]),
    "硬貨を寄せる(px)": (touching, [0, 60, 100, 130, 160]),
    "背景を灰色に": (background, [0, 0.3, 0.6, 0.8, 1.0]),
}


def evaluate(img):
    try:
        r = measure_image(img, ref_mm=REF_MM)
    except MarkerNotFoundError:
        return 0, float("nan")
    if r.count != len(TRUTH):
        return r.count, float("nan")
    measured = sorted(o.length_mm for o in r.objects)
    err = max(abs(m - t) / t for m, t in zip(measured, TRUTH)) * 100
    return r.count, err


def main():
    base = _read_image(ROOT / "tests" / "data" / "japanese_coins.jpg")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (fn, levels) in CONDITIONS.items():
        for lv in levels:
            img = fn(base, lv)
            count, err = evaluate(img)
            ok = count == len(TRUTH) and err <= 3.0
            rows.append([name, lv, count, "" if np.isnan(err) else f"{err:.1f}", "OK" if ok else "NG"])
            print(f"{name:12s} {lv:>6} -> {count} 個, 最大誤差 {err:5.1f}%  {'OK' if ok else 'NG'}")
            if lv == levels[-1] or not ok:
                cv2.imencode(".jpg", img)[1].tofile(str(OUT / f"{name}_{lv}.jpg"))
    with open(OUT / "robustness.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["条件", "強さ", "個数(正解5)", "最大誤差(%)", "判定"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
