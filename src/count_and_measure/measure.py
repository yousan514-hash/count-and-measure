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
    # 斜めのゆがみを補正したときは、補正後の画像（座標はこの画像の上の値）。補正しないときは None
    rectified: np.ndarray | None = field(default=None, repr=False)

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


def rectify_with_marker(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """マーカーが正方形に写るように画像を変形し、斜めから撮ったゆがみを直す。

    マーカーと物が同じ平面（机の上）にあることが前提。
    """
    side = _marker_side_px(corners)
    c0 = corners[0]
    square = np.float32([c0, c0 + [side, 0], c0 + [side, side], c0 + [0, side]])
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), square)
    h, w = image.shape[:2]
    frame = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(frame, H).reshape(-1, 2)
    x0, y0 = warped.min(axis=0)
    x1, y1 = warped.max(axis=0)
    # 変形後の画像が大きくなりすぎないよう、元の2倍までに収める
    s = min(1.0, np.sqrt(2 * w * h / max(1.0, (x1 - x0) * (y1 - y0))))
    T = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1]], np.float64)
    size = (int((x1 - x0) * s), int((y1 - y0) * s))
    border = tuple(int(v) for v in np.median(_border_pixels(image).reshape(-1, image.shape[2] if image.ndim == 3 else 1), axis=0))
    return cv2.warpPerspective(image, T @ H, size, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _marker_side_px(corners: np.ndarray) -> float:
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides))


def _touches_border(contour: np.ndarray, w: int, h: int, margin: int = 2) -> bool:
    x, y, cw, ch = cv2.boundingRect(contour)
    return x <= margin or y <= margin or x + cw >= w - margin or y + ch >= h - margin


def _border_pixels(img: np.ndarray) -> np.ndarray:
    return np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])


def _flatten_lighting(gray: np.ndarray, background_is_bright: bool) -> np.ndarray:
    """照明のむら（片側だけ暗い・中央だけ明るい）をならす。

    小さく縮めた画像で、物より大きい窓で背景だけの明るさを推定し、その明るさで割る。
    """
    h, w = gray.shape
    scale = 200 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    k = max(3, (min(small.shape) // 3) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # 明るい背景なら closing（暗い物を消す）、暗い背景なら opening（明るい物を消す）
    op = cv2.MORPH_CLOSE if background_is_bright else cv2.MORPH_OPEN
    bg = cv2.morphologyEx(small, op, kernel)
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    flat = gray.astype(np.float32) / np.maximum(bg, 1) * 200
    return np.clip(flat, 0, 255).astype(np.uint8)


def _foreground_mask(gray: np.ndarray, marker_region: np.ndarray | None) -> np.ndarray:
    """物を白（255）、背景を黒（0）にした2値画像を返す。"""
    # 強いノイズに負けないよう、まずメディアンフィルタで点状のノイズを消す
    blur = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (5, 5), 0)
    # マーカーの黒い部分が大きいと、2値化の境目が暗い側に寄って銀色の物を見落とす。
    # 先にマーカーを背景の明るさで塗りつぶしてから境目を決める。
    if marker_region is not None:
        blur = blur.copy()
        cv2.fillConvexPoly(blur, marker_region, int(np.median(_border_pixels(blur))))
    # 背景が明るい（紙）か暗い（布）か。決まった明るさで決めると、暗い部屋の白い紙や、
    # 片側だけ影になった紙で間違える。そこで両方の前提で照明をならして2値化し、
    # 画像の縁がより「背景」として残るほうを選ぶ（物は縁にはあまりかからない）
    # ※ 明るい方向・暗い方向の両方に外れた所を物とする方法も試したが、白い紙の上の1円玉
    #   （背景との差が小さい）を見落として悪くなった（experiments/README.md の 3回目）
    best = None
    for bright in (True, False):
        flat = _flatten_lighting(blur, bright)
        mode = cv2.THRESH_BINARY_INV if bright else cv2.THRESH_BINARY
        _, m = cv2.threshold(flat, 0, 255, mode | cv2.THRESH_OTSU)
        border_fg = float(np.mean(_border_pixels(m) > 0))
        if best is None or border_fg < best[0]:
            best = (border_fg, flat, m)
    _, blur, mask = best
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 銀色の物は、光が当たると明るさだけでは背景と見分けにくい。
    # 輪郭（エッジ）を太らせて閉じた形を作り、明るさで分けた結果と合わせて補う
    median = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0, 0.66 * median) * 0.5), int(min(255, 1.33 * median) * 0.5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    # 太らせた分だけ細くして、元の大きさに戻す
    edge_shapes = cv2.erode(_fill_holes(edges), kernel, iterations=2)
    # ノイズで輪郭が画面中に出たときは、輪郭の結果を使わない
    if cv2.countNonZero(edge_shapes) < 0.6 * edge_shapes.size:
        mask = cv2.bitwise_or(mask, edge_shapes)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 5円玉や50円玉の穴、物の表面の模様のすき間を埋める
    return _fill_holes(mask)


def _split_touching(mask: np.ndarray, min_area_px: float) -> np.ndarray:
    """くっついた物を分ける（距離変換 + Watershed）。

    丸い物が2つ触れると、くびれのある1つの形になる。くびれのある形（凸包に対して面積が小さい）だけ、
    中心からの距離の山を種にして分ける。
    """
    pad = 2
    out = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if area < 2 * min_area_px or hull_area == 0 or area / hull_area > 0.93:
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi = np.zeros((h, w), np.uint8)
        cv2.drawContours(roi, [c - [x, y]], -1, 255, -1)
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        _, peaks = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
        n, markers = cv2.connectedComponents(peaks.astype(np.uint8))
        if n <= 2:  # 山が1つ（背景＋1）なら分けない
            continue
        markers = markers + 1  # 種: 2, 3, ...
        markers[(roi == 255) & (peaks == 0)] = 0  # まだ決まっていない所
        markers[roi == 0] = 1  # 背景
        cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers)
        split = np.where(markers > 1, 255, 0).astype(np.uint8)
        split = cv2.erode(split, np.ones((3, 3), np.uint8))  # 境界線で確実に切り離す
        region = out[y:y + h, x:x + w]
        region[roi == 255] = split[roi == 255]
    return out[pad:-pad, pad:-pad]


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
    rectify: bool = True,
) -> MeasureResult:
    """画像（BGR）の中の物を数えて測る。

    物差しは2通り:
    - ArUco マーカー（既定）: marker_mm はマーカーの黒い四角の1辺（mm）
    - 大きさがわかっている物: ref_mm にその物の長辺（硬貨なら直径）を入れ、ref で
      "largest"（一番大きい物）か "smallest"（一番小さい物）を選ぶ。マーカーがなくても測れる

    keep_edge=True にすると、画像の縁に触れている物も数える（ぎりぎりで切り抜いた画像向け）。
    rectify=True（既定）なら、マーカーの形から斜めに撮ったゆがみを直してから測る。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    rectified = None
    if ref_mm is None and rectify:
        corners0 = detect_marker(gray, dict_name)
        # マーカーがほぼ正方形に写っている（真上から撮った）ときは変形しない
        sides = [np.linalg.norm(corners0[i] - corners0[(i + 1) % 4]) for i in range(4)]
        if max(sides) / min(sides) > 1.02:
            rectified = rectify_with_marker(image, corners0)
            image = rectified
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    pad = 0
    if keep_edge:
        # 縁に背景色の余白を足して、縁に触れた物も閉じた輪郭にする
        pad = 20
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                  value=int(np.median(_border_pixels(gray))))

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

    h, w = gray.shape
    # くっついた物を分ける。物の大きさがわからないうちは、画像の 0.05% を目安にする
    split_min = (min_area_mm2 / mm_per_px**2) if ref_mm is None else h * w * 5e-4
    mask = _split_touching(mask, split_min)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    return MeasureResult(mm_per_px=mm_per_px, marker_corners=corners, objects=objects, rectified=rectified)


def annotate(image: np.ndarray, result: MeasureResult) -> np.ndarray:
    """結果を書き込んだ画像を返す。"""
    # ゆがみを補正したときは、補正後の画像に書き込む（座標がその画像の上の値なので）
    out = (result.rectified if result.rectified is not None else image).copy()
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
