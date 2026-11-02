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
    """硬貨を寄せて置いた（右下の1円玉を左に動かして10円玉に近づける）。px は動かす量。

    1円玉の形だけを切り出して動かし、元の場所は周りの紙の色で埋める（inpaint）。
    PDCA 2回目までは四角く切り取って貼っていたため、灰色の四角い継ぎ目が残っていた。
    """
    if px == 0:
        return img
    coin = (_coin_mask(img)[..., 0] > 0.5).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(coin)
    # 右下の硬貨（中心が一番右下にあるもの）を1円玉とする
    target = max(range(1, n), key=lambda i: cent[i][0] + cent[i][1])
    one = (labels == target).astype(np.uint8) * 255
    one = cv2.dilate(one, np.ones((7, 7), np.uint8))
    out = cv2.inpaint(img, one, 7, cv2.INPAINT_TELEA)
    M = np.float32([[1, 0, -px], [0, 1, 0]])
    moved_coin = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    moved_mask = cv2.warpAffine(one, M, (img.shape[1], img.shape[0])) > 0
    out[moved_mask] = moved_coin[moved_mask]
    return out


_COIN_MASK = None


def _coin_mask(img):
    """元の写真の硬貨の部分（PDCA 2回目の方法で、正しく5枚取れることを確認済み）。"""
    global _COIN_MASK
    if _COIN_MASK is None:
        from count_and_measure import measure as m

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw = m._foreground_mask(gray, None)
        # 画像の端に触れない大きな形（＝硬貨5枚）だけを残す。端の明るいむらを硬貨と間違えない
        h, w = gray.shape
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if not m._touches_border(c, w, h)]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[: len(TRUTH)]
        mask = np.zeros_like(raw)
        cv2.drawContours(mask, contours, -1, 255, -1)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        _COIN_MASK = cv2.GaussianBlur(mask.astype(np.float32) / 255, (9, 9), 0)[..., None]
    return _COIN_MASK


def background(img, target):
    """背景（紙）の明るさを target に下げる。灰色の机の上に置いたときを再現する。

    硬貨の部分はそのまま。背景は明るさの比率だけ下げるので、影や照明のむらは残る。
    1回目の実験では「明るさ 200 より上の画素だけ置き換える」作り方をしていて、背景が白と灰色の
    まだらになっていた（本物の机とは違う）。PDCA 3回目でこの作り方に直した。
    """
    coin = _coin_mask(img)
    paper = np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[coin[..., 0] < 0.01])
    darker = img.astype(np.float32) * (target / paper)
    mixed = img.astype(np.float32) * coin + darker * (1 - coin)
    return np.clip(mixed, 0, 255).astype(np.uint8)


CONDITIONS = {
    "傾き(度)": (tilt, [0, 10, 20, 30, 40]),
    "暗さ": (dark, [0, 0.3, 0.5, 0.7, 0.85]),
    "ぼけ": (blur, [0, 3, 7, 12, 20]),
    "ノイズ": (noise, [0, 10, 25, 40, 60]),
    "片側の影": (shadow, [0, 0.3, 0.5, 0.7, 0.85]),
    "硬貨を寄せる(px)": (touching, [0, 60, 100, 130, 160]),
    "背景の明るさ": (background, [225, 190, 170, 150, 120]),
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
    ok_count = 0
    for name, (fn, levels) in CONDITIONS.items():
        for lv in levels:
            img = fn(base, lv)
            count, err = evaluate(img)
            ok = count == len(TRUTH) and err <= 3.0
            ok_count += ok
            rows.append([name, lv, count, "" if np.isnan(err) else f"{err:.1f}", "OK" if ok else "NG"])
            print(f"{name:12s} {lv:>6} -> {count} 個, 最大誤差 {err:5.1f}%  {'OK' if ok else 'NG'}")
            if lv == levels[-1] or not ok:
                cv2.imencode(".jpg", img)[1].tofile(str(OUT / f"{name}_{lv}.jpg"))
    print(f"== OK {ok_count} / {sum(len(v[1]) for v in CONDITIONS.values())}")
    with open(OUT / "robustness.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["条件", "強さ", "個数(正解5)", "最大誤差(%)", "判定"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
