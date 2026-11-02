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
    marker_corners: np.ndarray | None  # 物差しに物を使ったときは None
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


def _touches_border(contour: np.ndarray, w: int, h: int, margin: int = 2) -> bool:
    x, y, cw, ch = cv2.boundingRect(contour)
    return x <= margin or y <= margin or x + cw >= w - margin or y + ch >= h - margin


def _foreground_mask(gray: np.ndarray, marker_region: np.ndarray | None) -> np.ndarray:
    """物を白（255）、背景を黒（0）にした2値画像を返す。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 画像の縁の明るさで、背景が明るい（紙）か暗い（布）かを決める
    border = np.concatenate([blur[0, :], blur[-1, :], blur[:, 0], blur[:, -1]])
    background_is_bright = np.median(border) > 127
    # マーカーの黒い部分が大きいと、2値化の境目が暗い側に寄って銀色の物を見落とす。
    # 先にマーカーを背景の明るさで塗りつぶしてから境目を決める。
    if marker_region is not None:
        blur = blur.copy()
        cv2.fillConvexPoly(blur, marker_region, int(np.median(border)))
    mode = cv2.THRESH_BINARY_INV if background_is_bright else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, mode | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 銀色の物は、光が当たると明るさだけでは背景と見分けにくい。
    # 輪郭（エッジ）を太らせて閉じた形を作り、明るさで分けた結果と合わせて補う
    median = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0, 0.66 * median) * 0.5), int(min(255, 1.33 * median) * 0.5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    # 太らせた分だけ細くして、元の大きさに戻す
    edge_shapes = cv2.erode(_fill_holes(edges), kernel, iterations=2)
    mask = cv2.bitwise_or(mask, edge_shapes)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 5円玉や50円玉の穴、物の表面の模様のすき間を埋める
    return _fill_holes(mask)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """白い領域の中にある黒い穴を埋める（画像の縁につながる黒は背景として残す）。"""
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    filled = mask.copy()
    cv2.floodFill(filled, flood, (0, 0), 255)
    # 縁から塗れなかった黒＝穴
    holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, holes)


def measure_image(
    image: np.ndarray,
    marker_mm: float = 50.0,
    dict_name: str = "4x4_50",
    min_area_mm2: float = 10.0,
    ref_mm: float | None = None,
    ref: str = "largest",
    keep_edge: bool = False,
) -> MeasureResult:
    """画像（BGR）の中の物を数えて測る。

    物差しは2通り:
    - ArUco マーカー（既定）: marker_mm はマーカーの黒い四角の1辺（mm）
    - 大きさがわかっている物: ref_mm にその物の長辺（硬貨なら直径）を入れ、ref で
      "largest"（一番大きい物）か "smallest"（一番小さい物）を選ぶ。マーカーがなくても測れる

    keep_edge=True にすると、画像の縁に触れている物も数える（ぎりぎりで切り抜いた画像向け）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    pad = 0
    if keep_edge:
        # 縁に背景色の余白を足して、縁に触れた物も閉じた輪郭にする
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        pad = 20
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(np.median(border)))

    if ref_mm is None:
        corners = detect_marker(gray, dict_name)
        mm_per_px = marker_mm / _marker_side_px(corners)
        # マーカーと、その周りの白い余白を物として数えないようにする
        center = corners.mean(axis=0)
        marker_region = ((corners - center) * 1.6 + center).astype(np.int32)
        mask = _foreground_mask(gray, marker_region)
        cv2.fillConvexPoly(mask, marker_region, 0)
    else:
        corners = None
        mask = _foreground_mask(gray, None)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    if pad:
        # 足した余白の分だけ座標を戻す
        contours = [c - pad for c in contours]
        h, w = h - 2 * pad, w - 2 * pad
    else:
        # 画像の縁に触れている輪郭は、写り切っていない物か背景のむらなので除く
        contours = [c for c in contours if not _touches_border(c, w, h)]

    if ref_mm is not None:
        # 大きさがまだわからないので、まずは画像の 0.05% より小さいゴミだけ除く
        contours = [c for c in contours if cv2.contourArea(c) >= h * w * 5e-4]
        if not contours:
            raise MarkerNotFoundError("物差しにする物が見つかりません。")
        pick = max if ref == "largest" else min
        ref_contour = pick(contours, key=cv2.contourArea)
        mm_per_px = ref_mm / max(cv2.minAreaRect(ref_contour)[1])

    min_area_px = min_area_mm2 / (mm_per_px**2)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    # 左上から右下へ、読む順に番号を振る
    row = max(50, h // 8)
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1] // row, cv2.boundingRect(c)[0]))

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
    if corners is not None and pad:
        corners = corners - pad
    return MeasureResult(mm_per_px=mm_per_px, marker_corners=corners, objects=objects)


def annotate(image: np.ndarray, result: MeasureResult) -> np.ndarray:
    """結果を書き込んだ画像を返す。"""
    out = image.copy()
    if result.marker_corners is not None:
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
