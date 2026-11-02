import numpy as np
import pytest

from count_and_measure import measure_image
from count_and_measure.make_samples import PX_PER_MM, make_sample
from count_and_measure.measure import MarkerNotFoundError


@pytest.mark.parametrize("seed", range(1, 11))
def test_count_and_size(seed):
    img, meta = make_sample(seed)
    result = measure_image(img, marker_mm=meta["marker_mm"])

    assert result.count == meta["count"]
    assert result.mm_per_px == pytest.approx(1 / PX_PER_MM, rel=0.01)

    # 中心が一番近いもの同士を対応させて、大きさの誤差を見る
    for truth in meta["objects"]:
        obj = min(result.objects, key=lambda o: np.hypot(o.center_px[0] - truth["x"], o.center_px[1] - truth["y"]))
        assert obj.length_mm == pytest.approx(truth["length_mm"], rel=0.05)
        assert obj.width_mm == pytest.approx(truth["width_mm"], rel=0.15, abs=0.6)


def test_no_marker():
    img = np.full((400, 400, 3), 230, np.uint8)
    with pytest.raises(MarkerNotFoundError):
        measure_image(img)
