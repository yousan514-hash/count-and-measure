"""動作確認用の合成画像を作る。答え（個数と大きさ）が JSON でわかっているので、精度を測れる。

紙の上に ArUco マーカー（50mm）と、日本の硬貨サイズの円や、ネジのような細長い物を置いた画像を作る。
照明のむら・ノイズ・物の回転を入れて、本物の写真に少し近づけている。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PX_PER_MM = 8.0
MARKER_MM = 50.0
CANVAS = (1600, 2200)  # 高さ, 幅（px）= 200 x 275 mm

# 硬貨の直径（mm）と色（BGR）
COINS = {
    "10円": (23.5, (60, 100, 170)),
    "100円": (22.6, (150, 150, 150)),
    "500円": (26.5, (90, 150, 180)),
}
# ネジのような細長い物（長さ mm, 太さ mm）
RODS = [(30.0, 5.0), (40.0, 6.0), (20.0, 4.0)]


def _place(rng, occupied, radius_px, h, w, margin=40):
    for _ in range(500):
        x = rng.uniform(margin + radius_px, w - margin - radius_px)
        y = rng.uniform(margin + radius_px, h - margin - radius_px)
        if all(np.hypot(x - ox, y - oy) > radius_px + orad + 25 for ox, oy, orad in occupied):
            occupied.append((x, y, radius_px))
            return x, y
    raise RuntimeError("物を置く場所が足りません")


def make_sample(seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    h, w = CANVAS

    # 紙: 少し黄色みのある白に、照明のむら（グラデーション）を入れる
    yy, xx = np.mgrid[0:h, 0:w]
    light = 1.0 - 0.12 * ((xx / w - rng.uniform(0.3, 0.7)) ** 2 + (yy / h - rng.uniform(0.3, 0.7)) ** 2)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (225, 232, 236)
    img *= light[..., None]

    occupied = []
    # マーカー（白い余白付き）を左上の近くに置く
    side = int(MARKER_MM * PX_PER_MM)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, side)
    quiet = side // 5
    mx, my = 80, 80
    img[my - quiet:my + side + quiet, mx - quiet:mx + side + quiet] = 250
    img[my:my + side, mx:mx + side] = marker[..., None]
    occupied.append((mx + side / 2, my + side / 2, side * 0.8))

    truth = []
    n_coins = int(rng.integers(4, 8))
    for _ in range(n_coins):
        name = rng.choice(list(COINS))
        d_mm, color = COINS[name]
        r = d_mm * PX_PER_MM / 2
        x, y = _place(rng, occupied, r, h, w)
        shade = rng.uniform(0.85, 1.05)
        cv2.circle(img, (int(round(x)), int(round(y))), int(round(r)), [c * shade for c in color], -1, cv2.LINE_AA)
        truth.append({"kind": str(name), "x": x, "y": y, "length_mm": d_mm, "width_mm": d_mm})

    n_rods = int(rng.integers(1, 4))
    for _ in range(n_rods):
        length, width = RODS[int(rng.integers(len(RODS)))]
        x, y = _place(rng, occupied, length * PX_PER_MM / 2, h, w)
        angle = float(rng.uniform(0, 180))
        box = cv2.boxPoints(((x, y), (length * PX_PER_MM, width * PX_PER_MM), angle))
        cv2.fillConvexPoly(img, box.astype(np.int32), (70, 70, 75), cv2.LINE_AA)
        truth.append({"kind": "ネジ", "x": x, "y": y, "length_mm": length, "width_mm": width})

    img += rng.normal(0, 4, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    meta = {"seed": seed, "px_per_mm": PX_PER_MM, "marker_mm": MARKER_MM, "count": len(truth), "objects": truth}
    return img, meta


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "samples")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        img, meta = make_sample(seed=i)
        ok, buf = cv2.imencode(".png", img)
        buf.tofile(str(out / f"sample_{i:02d}.png"))
        (out / f"sample_{i:02d}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"sample_{i:02d}.png: {meta['count']} 個")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
