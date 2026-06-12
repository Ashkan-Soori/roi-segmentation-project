#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline

In this project I used OpenCV as a tool, not as a solution.

OpenCV gives me basic operations like filtering, thresholding and morphology,
but it does not understand what the image contains.

All decisions in this pipeline — like choosing the threshold,
removing noise, or deciding which region matters — are made manually.

The idea was to build something that I can understand step by step,
not just run a function and trust the result.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


def log(msg):
    print(f"[INFO] {msg}")


def load_image(path):
    """
    First thing I do is check if the image actually exists.

    This might sound obvious, but skipping this check usually leads
    to confusing errors later.

    I prefer to fail early and clearly.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("Something went wrong while loading the image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(img):
    """
    I convert the image to grayscale because I don’t really need color here.

    The segmentation is based on intensity differences,
    not color patterns.

    This also makes the problem simpler.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def custom_otsu(gray):
    """
    Instead of directly calling OpenCV’s Otsu,
    I tried to implement it myself.

    Not because OpenCV can't do it,
    but because I wanted to understand how the threshold is chosen.

    So here I explicitly compute it from the histogram.
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


def clean_mask(mask):
    """
    The raw mask is usually messy.

    There are small dots, holes, and broken regions.

    So I clean it in two steps:
    - remove tiny noise
    - make regions more continuous

    This part is mostly trial-and-error until it looks reasonable.
    """

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def extract_largest(mask):
    """
    After cleaning, there might still be multiple regions.

    I assume that the main tissue is the largest one.

    So instead of keeping everything,
    I keep only the biggest connected component.
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


def invert_mask(mask):
    """
    Just for visualization.

    White background + black tissue is easier to read
    than the opposite.
    """
    return cv2.bitwise_not(mask)


def overlay_edges(img, mask):
    """
    At first I was coloring the whole region,
    but that hides too much detail.

    So instead I only draw the edges.

    This way I can still see the tissue
    and also understand where the boundary is.
    """

    edges = cv2.Canny(mask, 100, 200)

    overlay = img.copy()
    overlay[edges > 0] = [255, 0, 0]

    return overlay


def draw_bbox(img, mask):
    """
    This is just a quick way to show where the ROI is.

    It doesn’t add new information,
    but it makes the result easier to understand visually.
    """

    coords = np.column_stack(np.where(mask > 0))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img_copy = img.copy()
    cv2.rectangle(img_copy, (x_min, y_min),
                  (x_max, y_max), (0, 255, 0), 2)

    return img_copy


def analyze(mask):
    """
    I didn’t want the code to just output an image.

    So I added a simple metric:
    how much of the image is considered tissue.

    It’s not perfect, but it gives a quick idea
    if the segmentation makes sense.
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


def show(img, gray, raw, clean, final,
         overlay, bbox, t, area, ratio, quality):
    """
    I try to show everything in the order it happens.

    So anyone looking at it can follow the process
    from the original image to the final result.

    No overlapping text, no confusion.
    """

    plt.figure(figsize=(18, 5))

    titles = ["Original", "Gray", "Raw",
              "Clean", "Final ROI", "Overlay", "BBox"]

    images = [img, gray, raw, clean, final, overlay, bbox]

    for i in range(7):
        plt.subplot(1, 7, i + 1)

        if i in [0, 5, 6]:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap="gray")

        plt.title(titles[i])
        plt.axis("off")

    plt.text(
        10, 30,
        f"T: {t}\nArea: {area}\nRatio: {ratio:.2f}\nQuality: {quality}",
        color="red",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    img = load_image(args.image)
    gray = to_gray(img)

    t, raw = custom_otsu(gray)

    clean = clean_mask(raw)

    # important step: remove remaining noise
    clean, _ = extract_largest(clean)

    final, area = extract_largest(clean)

    clean = invert_mask(clean)
    final = invert_mask(final)

    ratio, quality = analyze(final)

    overlay = overlay_edges(img, final)
    bbox = draw_bbox(img, final)

    if args.show:
        show(img, gray, raw, clean, final,
             overlay, bbox, t, area, ratio, quality)


if __name__ == "__main__":
    main()


# In[ ]:




