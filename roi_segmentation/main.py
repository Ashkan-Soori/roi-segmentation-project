#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline

In this implementation, OpenCV is used as a supporting tool for image operations.

However, the core logic of the pipeline — including threshold selection,
region filtering, and result analysis — is implemented and controlled manually.

The objective is to avoid treating image processing as a black-box workflow,
and instead provide a transparent and structured approach.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Logging utility
# -----------------------------
def log(message):
    """
    A lightweight logging helper.

    The intention here is to keep track of the pipeline execution
    in a clear and readable way without introducing unnecessary complexity.
    """
    print(f"[INFO] {message}")


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    """
    Load the input image and convert it to RGB.

    An explicit check is performed before loading
    to ensure that invalid paths are handled early.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("The image could not be loaded.")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------
# Convert to grayscale
# -----------------------------
def to_gray(img):
    """
    Convert the image to grayscale.

    This reduces the problem to a single intensity channel,
    which simplifies the thresholding stage.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# -----------------------------
# Custom Otsu implementation
# -----------------------------
def custom_otsu(gray):
    """
    Instead of relying on OpenCV's built-in Otsu method,
    the threshold is computed explicitly.

    OpenCV is used only for histogram extraction and final thresholding,
    while the decision process itself is implemented manually.

    This ensures full transparency over how the threshold is selected.
    """

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = gray.size

    best_threshold = 0
    max_variance = 0

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

        variance = bg_weight * fg_weight * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    _, mask = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)

    return best_threshold, mask


# -----------------------------
# Adaptive threshold selection
# -----------------------------
def choose_method(gray):
    """
    Introduce a data-driven decision instead of applying a fixed method.

    The choice between Otsu and a fixed threshold
    is based on the intensity distribution of the image.
    """

    std = np.std(gray)

    if std < 25:
        log("Low contrast detected → using manual threshold")
        return "manual"
    else:
        log("Sufficient contrast → using Otsu")
        return "otsu"


# -----------------------------
# Morphological refinement
# -----------------------------
def clean_mask(mask, kernel_size=3):
    """
    Refine the mask using morphological operations.

    Opening removes small isolated regions,
    while closing improves structural continuity.
    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return cleaned


# -----------------------------
# Remove small regions
# -----------------------------
def remove_small_objects(mask, min_size=1000):
    """
    Remove connected components below a certain size.

    This introduces a structural constraint,
    ensuring that only meaningful regions are preserved.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_size:
            cleaned[labels == i] = 255

    return cleaned


# -----------------------------
# Extract dominant region
# -----------------------------
def extract_largest(mask):
    """
    Select the dominant connected component.

    The assumption is that the primary tissue region
    corresponds to the largest area.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    largest_label = 1
    largest_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = i

    final = np.zeros_like(mask)
    final[labels == largest_label] = 255

    return final, largest_area


# -----------------------------
# Quantitative evaluation
# -----------------------------
def analyze_mask(mask):
    """
    Evaluate the segmentation outcome.

    The ratio of ROI pixels provides a quantitative indicator
    of the segmentation quality.
    """

    total = mask.size
    roi_pixels = np.sum(mask == 255)

    ratio = roi_pixels / total

    log(f"ROI ratio: {ratio:.2f}")

    if ratio < 0.05:
        print("[WARNING] The detected region is very small")

    return ratio


# -----------------------------
# Visualization
# -----------------------------
def show_results(img, gray, raw, clean, final, threshold, area, ratio):
    """
    Display intermediate and final results.

    Visualization is used as an analytical tool,
    with embedded numerical information for interpretation.
    """

    plt.figure(figsize=(16, 5))

    titles = ["Original", "Gray", "Raw", "Clean", "Final ROI"]
    images = [img, gray, raw, clean, final]

    for i in range(5):
        plt.subplot(1, 5, i + 1)

        if i == 0:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap="gray")

        plt.title(titles[i])
        plt.axis("off")

    info = f"Threshold: {threshold}\nArea: {area}\nRatio: {ratio:.2f}"

    plt.figtext(0.5, -0.05, info, ha="center",
                bbox={"facecolor": "white", "alpha": 0.8})

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    """
    Structured segmentation workflow:

    loading → transformation → segmentation → refinement → evaluation → output

    Each step is explicitly defined to maintain control over the process.
    """

    parser = argparse.ArgumentParser(description="ROI segmentation pipeline")

    parser.add_argument("--image", required=True)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    log("Loading image...")
    img = load_image(args.image)

    log("Converting to grayscale...")
    gray = to_gray(img)

    method = choose_method(gray)

    if method == "otsu":
        log("Applying custom Otsu...")
        t, raw = custom_otsu(gray)
    else:
        t = 180
        log("Applying manual threshold")
        _, raw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    log(f"Threshold value: {t}")

    log("Refining mask...")
    clean = clean_mask(raw, args.kernel)

    log("Filtering regions...")
    clean = remove_small_objects(clean)

    log("Extracting dominant region...")
    final, area = extract_largest(clean)

    ratio = analyze_mask(final)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "final_mask.png")

    cv2.imwrite(output_path, final)
    log(f"Result saved to {output_path}")

    if args.show:
        show_results(img, gray, raw, clean, final, t, area, ratio)

    log("Pipeline completed.")


if __name__ == "__main__":
    main()


# In[ ]:




