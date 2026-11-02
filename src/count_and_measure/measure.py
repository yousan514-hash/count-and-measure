"""ArUco マーカーを物差しにして、写真の中の物を数えて大きさを測る。

流れ:
1. ArUco マーカーを見つけ、1辺の長さ（既知の mm）から「1ピクセル = 何 mm」を求める
2. 背景と物を2値化で分ける（背景が明るいか暗いかは自動で判定）
3. マーカーの部分を除き、輪郭ごとに数と大きさを出す
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass
class MeasuredObject:
    index: int
    center_px: tuple[float, float]
    area_mm2: float
    equiv_diameter_mm: float
    length_mm: float  # 外接する回転長方形の長辺
    width_mm: float  # 同じく短辺
    contour: np.ndarray = field(repr=False)


@dataclass
class MeasureResult:
    mm_per_px: float
    marker_corners: np.ndarray
    objects: list[MeasuredObject]

    @property
    def count(self) -> int:
        return len(self.objects)


class MarkerNotFoundError(RuntimeError):
    pass


def detect_marker(gray: np.ndarray, dict_name: str = "4x4_50") -> np.ndarray:
    """最初に見つかった ArUco マーカーの4隅（4x2, px）を返す。"""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        raise MarkerNotFoundError("ArUco マーカーが見つかりません。マーカー全体が写っているか確認してください。")
    return corners[0].reshape(4, 2)


def _marker_side_px(corners: np.ndarray) -> float:
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides))


def _foreground_mask(gray: np.ndarray, marker_region: np.ndarray) -> np.ndarray:
    """物を白（255）、背景を黒（0）にした2値画像を返す。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 画像の縁の明るさで、背景が明るい（紙）か暗い（布）かを決める
    border = np.concatenate([blur[0, :], blur[-1, :], blur[:, 0], blur[:, -1]])
    background_is_bright = np.median(border) > 127
    # マーカーの黒い部分が大きいと、2値化の境目が暗い側に寄って銀色の物を見落とす。
    # 先にマーカーを背景の明るさで塗りつぶしてから境目を決める。
    blur = blur.copy()
    cv2.fillConvexPoly(blur, marker_region, int(np.median(border)))
    mode = cv2.THRESH_BINARY_INV if background_is_bright else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, mode | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def measure_image(
    image: np.ndarray,
    marker_mm: float = 50.0,
    dict_name: str = "4x4_50",
    min_area_mm2: float = 10.0,
) -> MeasureResult:
    """画像（BGR）の中の物を数えて測る。marker_mm はマーカーの黒い四角の1辺（mm）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners = detect_marker(gray, dict_name)
    mm_per_px = marker_mm / _marker_side_px(corners)

    # マーカーと、その周りの白い余白を物として数えないようにする
    center = corners.mean(axis=0)
    marker_region = ((corners - center) * 1.6 + center).astype(np.int32)
    mask = _foreground_mask(gray, marker_region)
    cv2.fillConvexPoly(mask, marker_region, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_px = min_area_mm2 / (mm_per_px**2)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    # 左上から右下へ、読む順に番号を振る
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1] // 50, cv2.boundingRect(c)[0]))

    objects = []
    for i, c in enumerate(contours, start=1):
        area_px = cv2.contourArea(c)
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        objects.append(
            MeasuredObject(
                index=i,
                center_px=(float(cx), float(cy)),
                area_mm2=area_px * mm_per_px**2,
                equiv_diameter_mm=float(np.sqrt(4 * area_px / np.pi)) * mm_per_px,
                length_mm=max(w, h) * mm_per_px,
                width_mm=min(w, h) * mm_per_px,
                contour=c,
            )
        )
    return MeasureResult(mm_per_px=mm_per_px, marker_corners=corners, objects=objects)


def annotate(image: np.ndarray, result: MeasureResult) -> np.ndarray:
    """結果を書き込んだ画像を返す。"""
    out = image.copy()
    cv2.polylines(out, [result.marker_corners.astype(np.int32)], True, (255, 0, 255), 3)
    for obj in result.objects:
        cv2.drawContours(out, [obj.contour], -1, (0, 200, 0), 3)
        x, y = int(obj.center_px[0]), int(obj.center_px[1])
        label = f"#{obj.index} {obj.length_mm:.1f}x{obj.width_mm:.1f}mm"
        cv2.putText(out, label, (x - 70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, label, (x - 70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"count: {result.count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(out, f"count: {result.count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 255), 3, cv2.LINE_AA)
    return out
