"""印刷用の ArUco マーカーを作る（A4・300dpi の PNG）。

印刷するときは「実際のサイズ（100%）」を選ぶこと。印刷したら定規で黒い四角の1辺を測り、
50mm からずれていたら、その値を --marker-mm に渡す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

DPI = 300
A4_MM = (297, 210)  # 縦, 横


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def make_marker_sheet(marker_mm: float = 50.0, marker_id: int = 0) -> np.ndarray:
    h, w = mm_to_px(A4_MM[0]), mm_to_px(A4_MM[1])
    sheet = np.full((h, w), 255, np.uint8)
    side = mm_to_px(marker_mm)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    y0, x0 = mm_to_px(30), (w - side) // 2
    sheet[y0:y0 + side, x0:x0 + side] = marker

    # 切り取り線（マーカーの周りに 10mm の白い余白を残す）
    pad = mm_to_px(10)
    cv2.rectangle(sheet, (x0 - pad, y0 - pad), (x0 + side + pad, y0 + side + pad), 160, 2)
    # 確認用の 100mm の目盛り線
    ty = y0 + side + mm_to_px(30)
    tx0, tx1 = (w - mm_to_px(100)) // 2, (w + mm_to_px(100)) // 2
    cv2.line(sheet, (tx0, ty), (tx1, ty), 0, 3)
    for i in range(11):
        x = tx0 + mm_to_px(10 * i)
        cv2.line(sheet, (x, ty - 25), (x, ty + 25), 0, 3)
    cv2.putText(sheet, f"ArUco 4x4_50 id={marker_id}  side={marker_mm:g}mm", (x0 - pad, y0 - pad - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
    cv2.putText(sheet, "100 mm (print at 100% / actual size)", (tx0, ty + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
    return sheet


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "marker_50mm_a4.png")
    sheet = make_marker_sheet()
    ok, buf = cv2.imencode(".png", sheet)
    buf.tofile(str(out))
    print(f"{out}: A4 / {DPI}dpi。実際のサイズ（100%）で印刷してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
