#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ROI Segmentation Pipeline (Manual + Coherence-Based Evaluation)

This version improves the evaluation step by replacing the simple
ratio-based quality metric with a coherence-based approach.

Instead of relying only on how much of the image is segmented,
we evaluate how structurally consistent the segmented region is.

This makes the interpretation more meaningful and closer to
real-world segmentation assessment.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    """Load image using PIL and convert to NumPy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


# -----------------------------
# Convert to grayscale
# -----------------------------
def to_gray(img):
    """
    Convert RGB to grayscale using perceptual weights.
    This removes color and keeps intensity information.
    """
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray.astype(np.uint8)


# -----------------------------
# Compute threshold (Otsu-style)
# -----------------------------
def compute_threshold(gray):
    """
    Compute threshold by maximizing inter-class variance.
    This separates foreground and background automatically.
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
    """Convert grayscale image into binary mask."""
    return np.where(gray <= t, 0, 255).astype(np.uint8)


# -----------------------------
# Dilation
# -----------------------------
def dilation(mask):
    """Expand foreground regions."""
    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if np.any(mask[i-1:i+2, j-1:j+2] == 0):
                output[i,j] = 0
            else:
                output[i,j] = 255

    return output


# -----------------------------
# Erosion
# -----------------------------
def erosion(mask):
    """Shrink foreground regions."""
    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if np.all(mask[i-1:i+2, j-1:j+2] == 0):
                output[i,j] = 0
            else:
                output[i,j] = 255

    return output


# -----------------------------
# Refinement
# -----------------------------
def refine_segmentation(mask):
    """
    Improve mask coherence using multiple morphology steps.
    """
    mask = dilation(mask)
    mask = dilation(mask)
    mask = erosion(mask)
    return mask


# -----------------------------
# Select main region
# -----------------------------
def select_main_region(mask):
    """
    Keep only the largest connected component.
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

            if visited[i,j] or mask[i,j] != 0:
                continue

            visited[i,j] = True
            region.append((i,j))

            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    stack.append((i+dx, j+dy))

        return region

    for i in range(h):
        for j in range(w):
            if not visited[i,j] and mask[i,j] == 0:
                region = dfs(i, j)
                if len(region) > len(best_region):
                    best_region = region

    output = np.full_like(mask, 255)
    for i, j in best_region:
        output[i,j] = 0

    return output


# -----------------------------
# Analyze (Coherence-based)
# -----------------------------
def analyze(mask):
    """
    Evaluate segmentation quality based on structural coherence.

    Idea:
    - A good segmentation should be mostly one connected region
    - Fragmented masks indicate poor quality

    We measure:
    - size of largest region
    - total foreground size
    """

    total_pixels = mask.size
    foreground_pixels = np.sum(mask == 0)

    ratio = foreground_pixels / total_pixels

    # Count connected components again (simple)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions = []

    def dfs(x, y):
        stack = [(x, y)]
        count = 0

        while stack:
            i, j = stack.pop()

            if i < 0 or i >= h or j < 0 or j >= w:
                continue

            if visited[i,j] or mask[i,j] != 0:
                continue

            visited[i,j] = True
            count += 1

            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    stack.append((i+dx, j+dy))

        return count

    for i in range(h):
        for j in range(w):
            if not visited[i,j] and mask[i,j] == 0:
                regions.append(dfs(i, j))

    if len(regions) == 0:
        return ratio, "Low"

    largest = max(regions)
    coherence = largest / foreground_pixels

    # final decision
    if coherence > 0.9:
        quality = "High"
    elif coherence > 0.6:
        quality = "Medium"
    else:
        quality = "Low"

    return ratio, quality


# -----------------------------
# Overlay
# -----------------------------
def create_overlay(img, mask):
    """Highlight boundaries manually."""
    overlay = img.copy()
    h, w = mask.shape

    for i in range(1, h-1):
        for j in range(1, w-1):
            if mask[i,j] == 0:
                if np.any(mask[i-1:i+2, j-1:j+2] == 255):
                    overlay[i,j] = [255, 0, 0]

    return overlay


# -----------------------------
# Visualization
# -----------------------------
def show_results(img, gray, mask, refined, final, t, ratio, quality):
    overlay = create_overlay(img, final)

    plt.figure(figsize=(18, 5))

    titles = ["Original", "Gray", "Mask", "Refined", "Final ROI", "Overlay"]
    images = [img, gray, mask, refined, final, overlay]

    for i in range(6):
        ax = plt.subplot(1, 6, i+1)

        if i in [0, 5]:
            ax.imshow(images[i])
        else:
            ax.imshow(images[i], cmap="gray")

        ax.set_title(titles[i])
        ax.axis("off")

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

    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        exit()

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


# In[ ]:




