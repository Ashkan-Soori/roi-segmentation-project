#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline

In this implementation, OpenCV is used as a computational framework
to support image processing operations.

However, the segmentation process itself is not delegated to predefined routines.

All key stages — including threshold estimation, pixel classification,
region validation, and region selection — are explicitly defined and controlled.

The objective is to construct a fully interpretable pipeline,
where each step reflects a deliberate design decision.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


def log(msg):
    print(f"[INFO] {msg}")


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    """
    The input is validated explicitly before processing.

    This ensures that any issue related to file access
    is handled early and clearly.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Unable to load image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------
# Convert to grayscale
# -----------------------------
def to_gray(img):
    """
    The segmentation relies on intensity distribution.

    Converting to grayscale allows the analysis
    to focus on structural variations rather than color.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# -----------------------------
# Threshold estimation
# -----------------------------
def compute_threshold(gray):
    """
    The threshold is derived explicitly from the image histogram.

    This ensures that the separation between foreground and background
    is determined by a controlled and interpretable criterion.
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

        variance = bg_weight * fg_weight * (mean_bg - mean_fg) ** 2

        if variance > max_var:
            max_var = variance
            best_t = t

    return int(best_t)


# -----------------------------
# Explicit threshold application
# -----------------------------
def apply_threshold(gray, t):
    """
    Pixel classification is performed explicitly.

    Each pixel is evaluated according to a defined rule:
    darker intensities correspond to the region of interest.
    """

    mask = np.zeros_like(gray, dtype=np.uint8)

    mask[gray < t] = 255

    return mask


# -----------------------------
# Refinement
# -----------------------------
def refine_segmentation(mask):
    """
    The segmentation is refined to improve structural consistency.

    This step removes isolated artifacts and reinforces continuity
    within the detected regions.
    """

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# -----------------------------
# Region validation
# -----------------------------
def is_valid_region(area, image_size):
    """
    A region is considered meaningful only if it satisfies
    a size constraint relative to the image.

    This introduces an explicit validation criterion.
    """

    return area > 0.01 * image_size


# -----------------------------
# Region selection
# -----------------------------
def select_main_region(mask):
    """
    Regions are evaluated using a defined validation rule.

    Among valid regions, the dominant one is selected
    as the final representation of the tissue.
    """

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    best_label = 1
    max_area = 0

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]

        if is_valid_region(area, mask.size):
            if area > max_area:
                max_area = area
                best_label = i

    final = np.zeros_like(mask)
    final[labels == best_label] = 255

    return final, max_area


# -----------------------------
# Visualization enhancement
# -----------------------------
def invert_mask(mask):
    """
    The representation is adjusted to improve visual clarity.

    The region of interest is shown in dark contrast
    against a bright background.
    """
    return cv2.bitwise_not(mask)


# -----------------------------
# Edge-based overlay
# -----------------------------
def overlay_edges(img, mask):
    """
    The boundary of the segmented region is highlighted.

    This preserves the original image content
    while clearly indicating the segmentation result.
    """

    edges = cv2.Canny(mask, 100, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    overlay = img.copy()
    overlay[edges > 0] = [255, 0, 0]

    return overlay


# -----------------------------
# Quantitative analysis
# -----------------------------
def analyze(mask):
    """
    The segmentation is interpreted using a quantitative measure.

    The proportion of detected region provides insight
    into the consistency of the result.
    """

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
# Visualization
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




