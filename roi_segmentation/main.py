#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ROI Segmentation Pipeline (Fully Manual + Display Fix)

This implementation avoids OpenCV entirely and builds the segmentation
pipeline step by step using NumPy.

Additionally, since the code is executed in a Windows CMD environment,
a matplotlib backend (TkAgg) is explicitly set to ensure that images
are displayed correctly.

Each stage is written explicitly to make the logic transparent:
- image loading
- grayscale conversion
- threshold computation
- segmentation
- refinement
- region extraction
- analysis
- visualization (with overlay)

The focus here is not only on results, but on understanding how each
step contributes to the final output.
"""

import matplotlib
matplotlib.use('TkAgg')   # 🔥 ensures display works in CMD

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    """
    Load the image using PIL instead of OpenCV.

    The image is converted into RGB format and then into a NumPy array
    so that we can process it numerically.
    """

    img = Image.open(path).convert("RGB")
    return np.array(img)


# -----------------------------
# Convert to grayscale
# -----------------------------
def to_gray(img):
    """
    Convert RGB image to grayscale manually.

    The weights (0.299, 0.587, 0.114) reflect how humans perceive brightness.
    This step simplifies the image by removing color information.
    """

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


# -----------------------------
# Compute threshold
# -----------------------------
def compute_threshold(gray):
    """
    Compute threshold using Otsu-style logic.

    The idea is to separate foreground and background by maximizing
    the variance between the two groups.
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
    Create binary mask.

    Pixels below threshold → foreground (0)
    Pixels above threshold → background (255)
    """

    return np.where(gray < t, 0, 255).astype(np.uint8)


# -----------------------------
# Dilation
# -----------------------------
def dilation(mask):
    """
    Expand foreground region.

    If any neighbor is foreground, the pixel becomes foreground.
    """

    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if np.any(mask[i-1:i+2, j-1:j+2] == 0):
                output[i, j] = 0
            else:
                output[i, j] = 255

    return output


# -----------------------------
# Erosion
# -----------------------------
def erosion(mask):
    """
    Shrink foreground region.

    A pixel remains foreground only if all neighbors are foreground.
    """

    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if np.all(mask[i-1:i+2, j-1:j+2] == 0):
                output[i, j] = 0
            else:
                output[i, j] = 255

    return output


# -----------------------------
# Refinement
# -----------------------------
def refine_segmentation(mask):
    """
    Improve segmentation quality.

    Closing (dilation + erosion) helps fill gaps and smooth boundaries.
    """

    return erosion(dilation(mask))


# -----------------------------
# Select main region
# -----------------------------
def select_main_region(mask):
    """
    Detect connected regions and select the largest one.

    This assumes the main object is the largest connected component.
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
# Analyze
# -----------------------------
def analyze(mask):
    """
    Compute ratio of foreground and assign a simple quality label.
    """

    ratio = np.sum(mask == 0) / mask.size

    if ratio > 0.7:
        quality = "High"
    elif ratio > 0.3:
        quality = "Medium"
    else:
        quality = "Low"

    return ratio, quality


# -----------------------------
# Overlay
# -----------------------------
def create_overlay(img, mask):
    """
    Highlight boundaries manually (no edge detector).

    A pixel is marked if it belongs to foreground but has
    background neighbors.
    """

    overlay = img.copy()
    h, w = mask.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if mask[i, j] == 0:
                if np.any(mask[i-1:i+2, j-1:j+2] == 255):
                    overlay[i, j] = [255, 0, 0]

    return overlay


# -----------------------------
# Visualization
# -----------------------------
def show_results(img, gray, mask, refined, final, t, ratio, quality):
    """
    Display all pipeline stages and overlay results.

    This makes it easy to visually inspect how segmentation evolves.
    """

    overlay = create_overlay(img, final)

    plt.figure(figsize=(18, 5))

    titles = ["Original", "Gray", "Mask", "Refined", "Final ROI", "Overlay"]
    images = [img, gray, mask, refined, final, overlay]

    for i in range(6):
        ax = plt.subplot(1, 6, i + 1)

        if i in [0, 5]:
            ax.imshow(images[i])
        else:
            ax.imshow(images[i], cmap="gray")

        ax.set_title(titles[i])
        ax.axis("off")

    # Display metrics on Final ROI
    ax = plt.subplot(1, 6, 5)
    ax.text(
        0.02, 0.95,
        f"T: {t}\nRatio: {ratio:.2f}\nQuality: {quality}",
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
if __name__ == "__main__":
    import sys

    image_path = sys.argv[1]

    img = load_image(image_path)
    gray = to_gray(img)

    t = compute_threshold(gray)

    mask = apply_threshold(gray, t)
    refined = refine_segmentation(mask)
    final = select_main_region(refined)

    ratio, quality = analyze(final)

    print(f"Threshold: {t}")
    print(f"Ratio: {ratio:.2f}")
    print(f"Quality: {quality}")

    show_results(img, gray, mask, refined, final, t, ratio, quality)

