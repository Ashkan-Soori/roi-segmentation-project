#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline

In this implementation, I tried to keep the entire process consistent
from beginning to end.

One important decision I made was to define the region of interest
as black (0) and the background as white (255), and to keep this
definition unchanged across all steps.

This avoids confusion and makes the pipeline easier to follow.
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
    """
    Before doing anything, I check that the image actually exists.

    It helps catch mistakes early instead of failing silently later.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Unable to read the image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------
# Convert to grayscale
# -----------------------------
def to_gray(img):
    """
    The segmentation is based on intensity, not color.

    So I convert the image to grayscale to simplify the problem.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# -----------------------------
# Compute threshold (manual Otsu)
# -----------------------------
def compute_threshold(gray):
    """
    I compute the threshold manually instead of calling a built-in function.

    This way, the decision process is fully visible and controlled.
    """

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
# Apply threshold (adaptive)
# -----------------------------
def apply_threshold(gray, t):
    """
    I don't assume which side of the threshold corresponds to tissue.

    So I create two candidates and evaluate them based on structure.
    """

    # Candidate 1: darker pixels = tissue
    mask_dark = np.full_like(gray, 255, dtype=np.uint8)
    mask_dark[gray < t] = 0

    # Candidate 2: brighter pixels = tissue
    mask_bright = np.full_like(gray, 255, dtype=np.uint8)
    mask_bright[gray > t] = 0

    def largest_component_area(mask):
        binary = np.where(mask == 0, 1, 0).astype(np.uint8)

        num, _, stats, _ = cv2.connectedComponentsWithStats(binary)

        if num <= 1:
            return 0

        return max(stats[1:, cv2.CC_STAT_AREA])

    area_dark = largest_component_area(mask_dark)
    area_bright = largest_component_area(mask_bright)

    if area_dark > area_bright:
        return mask_dark
    else:
        return mask_bright


# -----------------------------
# Refinement
# -----------------------------
def refine_segmentation(mask):
    """
    Here I improve the mask slightly.

    I only apply a gentle closing operation to fill small gaps,
    without changing the overall structure too much.

    Important: I keep the same representation:
    tissue = 0, background = 255
    """

    # temporary conversion for OpenCV (foreground must be white)
    temp = np.where(mask == 0, 255, 0).astype(np.uint8)

    kernel = np.ones((2, 2), np.uint8)

    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)

    # convert back to original format
    clean = np.where(temp == 255, 0, 255).astype(np.uint8)

    return clean


# -----------------------------
# Select main region
# -----------------------------
def select_main_region(mask):
    """
    Multiple regions may exist.

    Instead of just picking the largest blindly,
    I also consider how centrally located the region is.
    """

    binary = np.where(mask == 0, 1, 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    h, w = mask.shape

    best_label = 1
    best_score = 0

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        cx = x + width // 2
        cy = y + height // 2

        dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)

        score = area - 0.5 * dist

        if score > best_score:
            best_score = score
            best_label = i

    final = np.full_like(mask, 255)
    final[labels == best_label] = 0

    return final, best_score


# -----------------------------
# Overlay edges
# -----------------------------
def overlay_edges(img, mask):
    """
    I highlight only the boundary of the detected region.

    This makes the segmentation easier to interpret visually.
    """

    binary = np.where(mask == 0, 255, 0).astype(np.uint8)

    edges = cv2.Canny(binary, 100, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    overlay = img.copy()
    overlay[edges > 0] = [255, 0, 0]

    return overlay


# -----------------------------
# Analysis
# -----------------------------
def analyze(mask):
    """
    I compute how much of the image is considered tissue.

    This gives a quick idea of whether the segmentation looks reasonable.
    """

    total = mask.size
    roi = np.sum(mask == 0)

    ratio = roi / total

    if ratio > 0.7:
        quality = "High"
    elif ratio > 0.3:
        quality = "Medium"
    else:
        quality = "Low"

    return ratio, quality


# -----------------------------
# Visualization
# -----------------------------
def show(img, gray, raw, clean, final, overlay,
         t, area, ratio, quality):

    plt.figure(figsize=(18, 5))

    titles = ["Original", "Gray", "Raw",
              "Clean", "Final ROI", "Overlay"]

    images = [img, gray, raw, clean, final, overlay]

    for i in range(6):
        ax = plt.subplot(1, 6, i + 1)

        if i in [0, 5]:
            ax.imshow(images[i])
        else:
            ax.imshow(images[i], cmap="gray")

        ax.set_title(titles[i])
        ax.axis("off")

    # متن بالا سمت چپ بدون تداخل
    ax = plt.subplot(1, 6, 5)

    ax.text(
        0.02, 0.95,
        f"T: {t}\nArea: {int(area)}\nRatio: {ratio:.2f}\nQuality: {quality}",
        color="red",
        fontsize=10,
        verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.85)
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

    ratio, quality = analyze(final)
    overlay = overlay_edges(img, final)

    if args.show:
        show(img, gray, raw, clean, final,
             overlay, t, area, ratio, quality)


if __name__ == "__main__":
    main()


# In[ ]:




