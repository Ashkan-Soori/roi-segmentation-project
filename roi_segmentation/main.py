#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ROI Segmentation Pipeline (Manual Implementation)

In this implementation, I avoided relying on OpenCV and instead tried to
reconstruct the main steps of a segmentation pipeline using basic tools.

The goal here is not to optimize performance, but to understand how each
step works internally.

Each part of the pipeline is written explicitly:
- image loading
- grayscale conversion
- threshold computation
- segmentation
- refinement
- region selection
- simple analysis

This makes the behavior of the system transparent and easier to reason about.
"""

import numpy as np
from PIL import Image


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    """
    Instead of using OpenCV, I use PIL to load the image.

    The image is converted to RGB format and then to a NumPy array
    so that it can be processed numerically.
    """

    img = Image.open(path).convert("RGB")
    return np.array(img)


# -----------------------------
# Convert to grayscale
# -----------------------------
def to_gray(img):
    """
    Color information is not necessary for segmentation in this case.

    I convert the image to grayscale manually using a weighted sum
    of the RGB channels. These weights reflect human perception.
    """

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    return gray.astype(np.uint8)


# -----------------------------
# Compute threshold (Otsu-style)
# -----------------------------
def compute_threshold(gray):
    """
    The threshold is computed by analyzing the distribution of pixel intensities.

    The idea is to find a value that best separates the image into two groups:
    background and foreground.

    This is done by maximizing the variance between the two groups.
    """

    hist, _ = np.histogram(gray, bins=256, range=(0, 256))

    total = gray.size
    total_sum = np.dot(np.arange(256), hist)

    bg_sum = 0
    bg_weight = 0

    max_var = 0
    best_t = 0

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

    return best_t


# -----------------------------
# Apply threshold
# -----------------------------
def apply_threshold(gray, t):
    """
    Once the threshold is known, each pixel is classified.

    Pixels below the threshold are treated as foreground (0),
    while the rest are background (255).

    This creates a binary segmentation mask.
    """

    mask = np.where(gray < t, 0, 255)

    return mask.astype(np.uint8)


# -----------------------------
# Dilation (manual)
# -----------------------------
def dilation(mask):
    """
    Dilation expands the foreground region.

    A pixel becomes foreground if at least one of its neighbors
    is already part of the foreground.
    """

    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            neighborhood = mask[i-1:i+2, j-1:j+2]

            if np.any(neighborhood == 0):
                output[i, j] = 0
            else:
                output[i, j] = 255

    return output


# -----------------------------
# Erosion (manual)
# -----------------------------
def erosion(mask):
    """
    Erosion shrinks the foreground region.

    A pixel remains foreground only if all of its neighbors
    are also foreground.
    """

    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h - 1):
        for j in range(1, w - 1):

            neighborhood = mask[i-1:i+2, j-1:j+2]

            if np.all(neighborhood == 0):
                output[i, j] = 0
            else:
                output[i, j] = 255

    return output


# -----------------------------
# Refinement (closing)
# -----------------------------
def refine_segmentation(mask):
    """
    This step improves the segmentation result.

    I use a combination of dilation and erosion (closing)
    to fill small gaps and smooth the region boundaries.

    The goal is not to drastically change the shape,
    but to make the region more coherent.
    """

    return erosion(dilation(mask))


# -----------------------------
# Connected Components (manual)
# -----------------------------
def select_main_region(mask):
    """
    The segmentation may contain multiple regions.

    Here, I identify connected components manually
    and select the largest one, assuming it represents
    the main region of interest.
    """

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    best_region = []

    def dfs(x, y):
        stack = [(x, y)]
        region = []

        while stack:
            i, j = stack.pop()

            if i < 0 or i >= h or j < 0 or j >= w:
                continue

            if visited[i, j] or mask[i, j] != 0:
                continue

            visited[i, j] = True
            region.append((i, j))

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    stack.append((i + dx, j + dy))

        return region

    for i in range(h):
        for j in range(w):
            if not visited[i, j] and mask[i, j] == 0:
                region = dfs(i, j)

                if len(region) > len(best_region):
                    best_region = region

    output = np.full_like(mask, 255)

    for i, j in best_region:
        output[i, j] = 0

    return output


# -----------------------------
# Analysis
# -----------------------------
def analyze(mask):
    """
    This step provides a simple interpretation of the result.

    I compute how much of the image is classified as foreground.
    Based on that, I assign a rough quality label.
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

