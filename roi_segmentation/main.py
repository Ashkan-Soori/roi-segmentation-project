#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline

The segmentation is designed as a controlled and interpretable process.

Instead of assuming a fixed foreground/background relationship,
both possibilities are evaluated and the most consistent result is selected.

This ensures that the pipeline adapts to different image conditions
rather than relying on hard-coded assumptions.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Unable to load image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------
# Grayscale
# -----------------------------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# -----------------------------
# Custom Otsu
# -----------------------------
def compute_threshold(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    total = gray.size
    total_sum = np.dot(np.arange(256), hist)

    bg_sum = 0
    bg_weight = 0

    best_t = 0
    max_var = 0

    for t in range(256):
        bg_weight += hist[t]

        if bg_weight == 0:
            continue

        fg_weight = total - bg_weight
        if fg_weight == 0:
            break

        bg_sum += t * hist[t]

        mean_bg = bg_sum / bg_weight
        mean_fg = (total_sum - bg_sum) / fg_weight

        var = bg_weight * fg_weight * (mean_bg - mean_fg) ** 2

        if var > max_var:
            max_var = var
            best_t = t

    return int(best_t)


# -----------------------------
# Adaptive thresholding (FIXED)
# -----------------------------
def apply_threshold(gray, t):
    """
    Instead of assuming which side represents the tissue,
    both interpretations are evaluated.

    The version that produces a more consistent region
    (larger connected structure) is selected.
    """

    mask_dark = np.zeros_like(gray, dtype=np.uint8)
    mask_dark[gray < t] = 255

    mask_bright = np.zeros_like(gray, dtype=np.uint8)
    mask_bright[gray > t] = 255

    # choose the mask with more meaningful structure
    if np.sum(mask_dark) > np.sum(mask_bright):
        return mask_dark
    else:
        return mask_bright


# -----------------------------
# Refinement (LESS aggressive)
# -----------------------------
def refine_segmentation(mask):
    """
    The refinement step is intentionally moderate.

    Excessive filtering can remove valid structures,
    so the kernel size is kept small.
    """

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# -----------------------------
# Region selection (FIXED)
# -----------------------------
def select_main_region(mask):
    """
    Instead of applying a strict threshold on region size,
    the dominant region is selected directly.

    This avoids discarding valid tissue in cases
    where the segmented region is relatively small.
    """

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    largest = 1
    max_area = 0

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > max_area:
            max_area = area
            largest = i

    final = np.zeros_like(mask)
    final[labels == largest] = 255

    return final, max_area


# -----------------------------
# Visualization helpers
# -----------------------------
def invert_mask(mask):
    return cv2.bitwise_not(mask)


def overlay_edges(img, mask):
    """
    Boundaries are highlighted instead of full regions,
    preserving the original visual context.
    """

    edges = cv2.Canny(mask, 100, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    overlay = img.copy()
    overlay[edges > 0] = [255, 0, 0]

    return overlay


# -----------------------------
# Analysis
# -----------------------------
def analyze(mask):
    total = mask.size
    roi = np.sum(mask == 255)

    ratio = roi / total

    if ratio > 0.7:
        quality = "High"
    elif ratio > 0.3:
        quality = "Medium"
    else:
        quality = "Low"

    return ratio, quality


# -----------------------------
# Show
# -----------------------------
def show(img, gray, raw, clean, final, overlay,
         t, area, ratio, quality):

    plt.figure(figsize=(18, 5))

    titles = ["Original", "Gray", "Raw",
              "Clean", "Final ROI", "Overlay"]

    images = [img, gray, raw, clean, final, overlay]

    for i in range(6):
        plt.subplot(1, 6, i + 1)

        if i in [0, 5]:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap="gray")

        plt.title(titles[i])
        plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.text(
        10, 30,
        f"T: {t}\nArea: {area}\nRatio: {ratio:.2f}\nQuality: {quality}",
        color="red",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    img = load_image(args.image)
    gray = to_gray(img)

    t = compute_threshold(gray)

    raw = apply_threshold(gray, t)
    clean = refine_segmentation(raw)

    final, area = select_main_region(clean)

    clean_display = invert_mask(clean)
    final_display = invert_mask(final)

    ratio, quality = analyze(final)
    overlay = overlay_edges(img, final)

    if args.show:
        show(img, gray, raw, clean_display,
             final_display, overlay,
             t, area, ratio, quality)


if __name__ == "__main__":
    main()


# In[ ]:




