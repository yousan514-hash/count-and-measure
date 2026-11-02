"""コマンドラインから使う入口。

例:
    pixi run measure samples/sample_01.png --marker-mm 50
    pixi run measure photos/ --out results
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from .measure import ARUCO_DICTS, MarkerNotFoundError, annotate, measure_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_image(path: Path) -> np.ndarray | None:
    # cv2.imread は日本語のパスを読めないので、バイト列から読む
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, image)
    if ok:
        buf.tofile(str(path))


def _collect(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        else:
            paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="写真から物を数えて、大きさを mm で測る")
    parser.add_argument("inputs", nargs="+", help="画像ファイルかフォルダ")
    parser.add_argument("--marker-mm", type=float, default=50.0, help="ArUco マーカーの1辺（mm）。既定 50")
    parser.add_argument("--dict", default="4x4_50", choices=sorted(ARUCO_DICTS), help="マーカーの種類")
    parser.add_argument("--min-area-mm2", type=float, default=10.0, help="これより小さいものは無視する（mm²）")
    parser.add_argument("--out", default="results", help="結果の保存先フォルダ")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    failed = 0

    for path in _collect(args.inputs):
        image = _read_image(path)
        if image is None:
            print(f"[skip] 読めない画像: {path}", file=sys.stderr)
            failed += 1
            continue
        try:
            result = measure_image(image, args.marker_mm, args.dict, args.min_area_mm2)
        except MarkerNotFoundError as e:
            print(f"[skip] {path.name}: {e}", file=sys.stderr)
            failed += 1
            continue

        _write_image(out_dir / f"{path.stem}_result.png", annotate(image, result))
        with open(out_dir / f"{path.stem}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["番号", "中心x(px)", "中心y(px)", "長辺(mm)", "短辺(mm)", "円相当径(mm)", "面積(mm2)"])
            for o in result.objects:
                w.writerow([o.index, f"{o.center_px[0]:.0f}", f"{o.center_px[1]:.0f}",
                            f"{o.length_mm:.2f}", f"{o.width_mm:.2f}", f"{o.equiv_diameter_mm:.2f}", f"{o.area_mm2:.1f}"])
        summary_rows.append([path.name, result.count, f"{result.mm_per_px:.4f}"])
        print(f"{path.name}: {result.count} 個（1px = {result.mm_per_px:.4f} mm）")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["画像", "個数", "mm/px"])
        w.writerows(summary_rows)
    return 1 if failed and not summary_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
