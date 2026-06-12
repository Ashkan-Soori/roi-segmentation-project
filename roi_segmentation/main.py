#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline

In this implementation, OpenCV is used as a supporting tool for image operations.

However, the core logic of the pipeline — including threshold selection,
region filtering, and result analysis — is implemented and controlled manually.

The objective is to avoid treating image processing as a black-box workflow,
and instead provide a transparent, structured, and interpretable approach.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Logging
# -----------------------------
def log(msg):
    print(f"[INFO] {msg}")


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    """
    The pipeline starts here.

    Before doing anything fancy, I make sure the image actually exists.
    This avoids those frustrating silent errors later.

    Then I load it and convert it to RGB, because OpenCV uses BGR by default.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Failed to load image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------
# Grayscale
# -----------------------------
def to_gray(img):
    """
    Instead of working with 3 channels,
    I reduce the problem to a single intensity channel.

    This makes thresholding much more stable and interpretable.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# -----------------------------
# Custom Otsu
# -----------------------------
def custom_otsu(gray):
    """
    Rather than relying on OpenCV’s built-in Otsu,
    I explicitly compute the threshold.

    This gives full control over the decision process
    and avoids treating thresholding as a black box.
    """

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.size

    best_t = 0
    max_var = 0

    total_sum = np.dot(np.arange(256), hist)

    bg_sum = 0
    bg_weight = 0

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

    _, mask = cv2.threshold(gray, best_t, 255, cv2.THRESH_BINARY)

    return best_t, mask


# -----------------------------
# Adaptive method
# -----------------------------
def choose_method(gray):
    """
    I avoid applying a fixed method blindly.

    Instead, I look at the image statistics (standard deviation)
    and decide if Otsu is reliable.

    This introduces a small but meaningful layer of reasoning.
    """

    if np.std(gray) < 25:
        log("Low contrast → manual threshold")
        return "manual"
    else:
        log("Using Otsu")
        return "otsu"


# -----------------------------
# Morphology
# -----------------------------
def clean_mask(mask, k=3):
    """
    Raw segmentation is rarely clean.

    So I refine it:
    - Opening removes isolated noise
    - Closing improves structure continuity
    """

    kernel = np.ones((k, k), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# -----------------------------
# Remove noise regions
# -----------------------------
def remove_small_objects(mask, min_size=1000):
    """
    Not every detected region is meaningful.

    Here I filter out small components
    that are unlikely to be part of the main tissue.
    """

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    cleaned = np.zeros_like(mask)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > min_size:
            cleaned[labels == i] = 255

    return cleaned


# -----------------------------
# Largest ROI
# -----------------------------
def extract_largest(mask):
    """
    Among all regions, I assume the main tissue
    corresponds to the largest connected component.
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
# Region count
# -----------------------------
def count_regions(mask):
    """
    Counting regions helps understand segmentation complexity.
    """
    num, _, _, _ = cv2.connectedComponentsWithStats(mask)
    return num - 1


# -----------------------------
# Quality score
# -----------------------------
def quality_score(ratio):
    """
    A simple qualitative interpretation of ROI size.
    """

    if ratio > 0.7:
        return "High"
    elif ratio > 0.3:
        return "Medium"
    else:
        return "Low"


# -----------------------------
# Overlay mask
# -----------------------------
def overlay_mask(img, mask):
    """
    Instead of just showing the mask,
    I overlay it on the original image.

    This makes it much easier to visually verify
    if the segmentation aligns with the tissue.
    """

    overlay = img.copy()
    overlay[mask == 255] = [255, 0, 0]  # red highlight

    return overlay


# -----------------------------
# Bounding box
# -----------------------------
def draw_bbox(img, mask):
    """
    Draw a bounding box around the detected ROI.

    This gives a spatial summary of the segmentation.
    """

    coords = np.column_stack(np.where(mask > 0))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img_copy = img.copy()

    cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return img_copy


# -----------------------------
# Analysis
# -----------------------------
def analyze_mask(mask):
    """
    I don't want the pipeline to just produce an output.

    This step interprets the result by computing
    how much of the image is considered ROI.
    """

    total = mask.size
    roi = np.sum(mask == 255)

    ratio = roi / total

    log(f"ROI ratio: {ratio:.2f}")

    return ratio


# -----------------------------
# Visualization
# -----------------------------
def show_results(img, gray, raw, clean, final, overlay, boxed,
                 t, area, ratio, regions, quality):
    """
    This is not just visualization.

    It is a diagnostic view of the entire pipeline,
    showing both transformations and interpretation.
    """

    plt.figure(figsize=(18, 5))

    titles = ["Original", "Overlay", "Raw", "Clean", "Final ROI", "BBox"]
    images = [img, overlay, raw, clean, final, boxed]

    for i in range(6):
        plt.subplot(1, 6, i + 1)

        if i in [0, 1, 5]:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap="gray")

        plt.title(titles[i])
        plt.axis("off")

    # overlay text
    plt.subplot(1, 6, 5)

    text = (
        f"T: {t}\n"
        f"Area: {area}\n"
        f"Ratio: {ratio:.2f}\n"
        f"Regions: {regions}\n"
        f"Quality: {quality}"
    )

    plt.text(5, 20, text, color="red",
             bbox=dict(facecolor="white", alpha=0.7))

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

    method = choose_method(gray)

    if method == "otsu":
        t, raw = custom_otsu(gray)
    else:
        t = 180
        _, raw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    clean = clean_mask(raw)
    clean = remove_small_objects(clean)

    final, area = extract_largest(clean)

    ratio = analyze_mask(final)
    regions = count_regions(clean)
    quality = quality_score(ratio)

    overlay = overlay_mask(img, final)
    boxed = draw_bbox(img, final)

    if args.show:
        show_results(img, gray, raw, clean, final,
                     overlay, boxed,
                     t, area, ratio, regions, quality)


if __name__ == "__main__":
    main()


# In[ ]:




