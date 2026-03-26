"""
GrabCut-based document segmentation using GMM foreground/background models.

Uses OpenCV's GrabCut algorithm to separate document foreground from background
based on Gaussian Mixture Models. Better for documents with irregular/colored
backgrounds compared to threshold-based methods.

Key parameters (all PSO-tunable):
  - margin_ratio: shrink ratio for initial foreground rectangle
  - iter_count: GrabCut iteration count
  - morph_kernel_size: morphological closing kernel
  - min_area_ratio: minimum component area filter
  - padding: bounding box expansion
"""
import numpy as np
import cv2
from PIL import Image


def grabcut_segment(
    img: Image.Image,
    margin_ratio: float = 0.02,
    iter_count: int = 5,
    morph_kernel_size: int = 15,
    min_area_ratio: float = 0.001,
    padding: int = 10,
    mode: str = "whiten",
) -> Image.Image:
    """
    Segment document foreground using GrabCut algorithm.

    Parameters
    ----------
    img : PIL.Image
        Input document image (RGB).
    margin_ratio : float
        Fraction of image width/height used as margin for the initial rectangle.
    iter_count : int
        Number of GrabCut iterations.
    morph_kernel_size : int
        Morphological closing kernel size.
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
    h, w = arr.shape[:2]
    total_area = h * w

    # Initial rectangle: image with margins removed
    mx = max(1, int(w * margin_ratio))
    my = max(1, int(h * margin_ratio))
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(arr, mask, rect, bgd_model, fgd_model,
                    iter_count, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return img

    # Foreground = definite foreground + probable foreground
    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Morphological closing to merge nearby regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Connected component analysis to filter noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8
    )
    min_area = total_area * min_area_ratio

    clean_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    if clean_mask.sum() == 0:
        return img

    coords = cv2.findNonZero(clean_mask)
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
