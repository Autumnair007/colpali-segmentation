"""
Edge-based document segmentation using Canny edge detection + contour analysis.

Detects document content regions by finding areas with high edge density.
Works well for text-heavy documents where content regions have dense edges
compared to blank/background areas.

Key parameters (all PSO-tunable):
  - canny_low / canny_high: Canny hysteresis thresholds
  - dilate_kernel_size: dilation kernel to merge nearby edges
  - min_area_ratio: minimum component area filter
  - padding: bounding box expansion
"""
import numpy as np
import cv2
from PIL import Image


def edge_segment(
    img: Image.Image,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_kernel_size: int = 15,
    min_area_ratio: float = 0.001,
    padding: int = 10,
    mode: str = "whiten",
) -> Image.Image:
    """
    Segment document foreground using Canny edge detection and contour merging.

    Parameters
    ----------
    img : PIL.Image
        Input document image (RGB).
    canny_low : int
        Lower hysteresis threshold for Canny.
    canny_high : int
        Upper hysteresis threshold for Canny.
    dilate_kernel_size : int
        Kernel size for dilating edges to form connected regions.
    min_area_ratio : float
        Minimum connected component area as fraction of total area.
    padding : int
        Padding pixels around detected bounding box.
    mode : str
        'whiten' — set background to white, keep original size.
        'crop'   — crop to the bounding box of foreground regions.

    Returns
    -------
    PIL.Image
        Processed document image.
    """
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    total_area = h * w

    # Blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate edges to form connected regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size)
    )
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Morphological closing to fill gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    min_area = total_area * min_area_ratio

    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    if mask.sum() == 0:
        return img

    coords = cv2.findNonZero(mask)
    x, y, bw, bh = cv2.boundingRect(coords)

    x = max(0, x - padding)
    y = max(0, y - padding)
    bw = min(w - x, bw + 2 * padding)
    bh = min(h - y, bh + 2 * padding)

    if mode == "crop":
        return Image.fromarray(arr[y:y + bh, x:x + bw])

    result = arr.copy()
    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[y:y + bh, x:x + bw] = 255
    result[box_mask == 0] = 255
    return Image.fromarray(result)
