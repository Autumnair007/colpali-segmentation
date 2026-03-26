# tests/test_segmentation.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def document_image():
    """White background with dark document area in center."""
    arr = np.ones((400, 300, 3), dtype=np.uint8) * 240
    arr[50:350, 30:270] = 30
    return Image.fromarray(arr)


# ──────────────── Adaptive ────────────────

def test_adaptive_returns_pil(document_image):
    from robust.segmentation.adaptive_seg import adaptive_segment
    assert isinstance(adaptive_segment(document_image), Image.Image)


def test_adaptive_whiten_same_size(document_image):
    from robust.segmentation.adaptive_seg import adaptive_segment
    result = adaptive_segment(document_image, mode="whiten")
    assert result.size == document_image.size


def test_adaptive_crop_smaller_or_equal(document_image):
    from robust.segmentation.adaptive_seg import adaptive_segment
    result = adaptive_segment(document_image, mode="crop")
    assert result.size[0] <= document_image.size[0]
    assert result.size[1] <= document_image.size[1]


def test_adaptive_background_white(document_image):
    from robust.segmentation.adaptive_seg import adaptive_segment
    result = adaptive_segment(document_image, mode="whiten")
    arr = np.array(result)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    for corner in corners:
        assert corner.mean() > 200, f"Corner {corner} should be white"


def test_adaptive_handles_white_image():
    """All-white image should return unchanged (no crash)."""
    from robust.segmentation.adaptive_seg import adaptive_segment
    white = Image.new("RGB", (200, 200), (255, 255, 255))
    result = adaptive_segment(white)
    assert isinstance(result, Image.Image)


def test_adaptive_handles_black_image():
    """All-black image should not crash."""
    from robust.segmentation.adaptive_seg import adaptive_segment
    black = Image.new("RGB", (200, 200), (0, 0, 0))
    result = adaptive_segment(black)
    assert isinstance(result, Image.Image)


# ──────────────── GrabCut ────────────────

def test_grabcut_returns_pil(document_image):
    from robust.segmentation.grabcut_seg import grabcut_segment
    assert isinstance(grabcut_segment(document_image), Image.Image)


def test_grabcut_whiten_same_size(document_image):
    from robust.segmentation.grabcut_seg import grabcut_segment
    result = grabcut_segment(document_image, mode="whiten")
    assert result.size == document_image.size


def test_grabcut_crop_smaller_or_equal(document_image):
    from robust.segmentation.grabcut_seg import grabcut_segment
    result = grabcut_segment(document_image, mode="crop")
    assert result.size[0] <= document_image.size[0]
    assert result.size[1] <= document_image.size[1]


def test_grabcut_handles_white_image():
    from robust.segmentation.grabcut_seg import grabcut_segment
    white = Image.new("RGB", (200, 200), (255, 255, 255))
    result = grabcut_segment(white)
    assert isinstance(result, Image.Image)


def test_grabcut_handles_black_image():
    from robust.segmentation.grabcut_seg import grabcut_segment
    black = Image.new("RGB", (200, 200), (0, 0, 0))
    result = grabcut_segment(black)
    assert isinstance(result, Image.Image)


# ──────────────── Edge-based ────────────────

def test_edge_returns_pil(document_image):
    from robust.segmentation.edge_seg import edge_segment
    assert isinstance(edge_segment(document_image), Image.Image)


def test_edge_whiten_same_size(document_image):
    from robust.segmentation.edge_seg import edge_segment
    result = edge_segment(document_image, mode="whiten")
    assert result.size == document_image.size


def test_edge_crop_smaller_or_equal(document_image):
    from robust.segmentation.edge_seg import edge_segment
    result = edge_segment(document_image, mode="crop")
    assert result.size[0] <= document_image.size[0]
    assert result.size[1] <= document_image.size[1]


def test_edge_handles_white_image():
    from robust.segmentation.edge_seg import edge_segment
    white = Image.new("RGB", (200, 200), (255, 255, 255))
    result = edge_segment(white)
    assert isinstance(result, Image.Image)


def test_edge_handles_black_image():
    from robust.segmentation.edge_seg import edge_segment
    black = Image.new("RGB", (200, 200), (0, 0, 0))
    result = edge_segment(black)
    assert isinstance(result, Image.Image)
