#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline

In this script, I tried to approach the segmentation problem step by step,
instead of relying on a single built-in function.

The idea is to understand what is happening at each stage:
how the threshold is chosen, how pixels are classified,
and how the final region is selected.

The goal is not just to get a result, but to make the whole process clear and controllable.
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
    I start by checking if the file actually exists.

    It avoids confusing errors later and makes debugging easier.
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
    The segmentation is based on intensity values,
    so I convert the image to grayscale.

    This removes color complexity and keeps the focus on structure.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# -----------------------------
# Compute threshold (manual Otsu)
# -----------------------------
def compute_threshold(gray):
    """
    Instead of using a built-in function,
    I compute the threshold manually using Otsu's method.

    This helps me understand how the separation between foreground
    and background is actually decided.
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
# Apply threshold (adaptive)
# -----------------------------
def apply_threshold(gray, t):
    """
    I don't assume beforehand which side of the threshold is the tissue.

    So I create two possible interpretations and evaluate them
    based on how structured they are.

    Also, I define the final mask so that:
    - tissue is black (0)
    - background is white (255)
    """

    mask_dark = np.full_like(gray, 255, dtype=np.uint8)
    mask_dark[gray < t] = 0

    mask_bright = np.full_like(gray, 255, dtype=np.uint8)
    mask_bright[gray > t] = 0

    def largest_component(mask):
        binary = (mask == 0).astype(np.uint8)

        num, _, stats, _ = cv2.connectedComponentsWithStats(binary)

        if num <= 1:
            return 0

        return max(stats[1:, cv2.CC_STAT_AREA])

    area_dark = largest_component(mask_dark)
    area_bright = largest_component(mask_bright)

    if area_dark > area_bright:
        return mask_dark
    else:
        return mask_bright


# -----------------------------
# Refinement
# -----------------------------
def refine_segmentation(mask):
    """
    After thresholding, there are usually small gaps and noise.

    I apply a light refinement to make the region more continuous,
    without removing important details.
    """

    binary = (mask == 0).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    clean = np.full_like(mask, 255)
    clean[binary == 255] = 0

    return clean


# -----------------------------
# Select main region
# -----------------------------
def select_main_region(mask):
    """
    At this point, there might still be multiple regions.

    Instead of just picking the largest one,
    I also consider where the region is located.

    Regions that are both large and closer to the center
    are more likely to represent the actual tissue.
    """

    binary = (mask == 0).astype(np.uint8)

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
    Instead of coloring the whole region,
    I highlight only the boundaries.

    This keeps the original image visible
    while still showing the segmentation clearly.
    """

    binary = (mask == 0).astype(np.uint8) * 255

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
    I compute a simple ratio to understand how much
    of the image is considered tissue.

    This helps interpret whether the segmentation looks reasonable.
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

    # ✅ متن تحلیل بالا سمت چپ بدون تداخل
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




