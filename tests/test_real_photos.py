"""本物の写真（Wikimedia Commons、出典は ATTRIBUTION.md）で、硬貨を数えて直径を測る。

マーカーは写っていないので、一番大きい硬貨を物差しにする。
"""
from pathlib import Path

import pytest

from count_and_measure import measure_image
from count_and_measure.cli import _read_image

DATA = Path(__file__).parent / "data"

# 日本の硬貨の直径（mm）
DIAMETER = {"1円": 20.0, "5円": 22.0, "10円": 23.5, "50円": 21.0, "100円": 22.6, "500円": 26.5}


@pytest.mark.parametrize(
    "name, ref_mm, keep_edge, expected",
    [
        # 白い紙の上の実写。影と照明のむらがある。左上から読む順
        ("japanese_coins.jpg", DIAMETER["10円"], False, ["100円", "50円", "10円", "5円", "1円"]),
        # 切り抜き画像。硬貨が画像の縁ぎりぎりまである
        ("jpy_coins_2.png", DIAMETER["500円"], True, ["1円", "5円", "10円", "500円", "50円", "100円"]),
    ],
)
def test_coins(name, ref_mm, keep_edge, expected):
    image = _read_image(DATA / name)
    result = measure_image(image, ref_mm=ref_mm, keep_edge=keep_edge)

    assert result.count == len(expected)
    # 番号は行ごとに左から。2行目の並びは画像による
    measured = sorted(o.length_mm for o in result.objects)
    truth = sorted(DIAMETER[c] for c in expected)
    for m, t in zip(measured, truth):
        assert m == pytest.approx(t, rel=0.03)
