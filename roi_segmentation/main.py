#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ROI Segmentation Pipeline (Manual + Coherence-Based Evaluation)

In this project, I tried to build a simple image segmentation pipeline
from scratch, without using OpenCV or advanced libraries.

The idea is to understand each step instead of relying on built-in functions.

The pipeline is:
image → grayscale → threshold → clean → main region → analysis → visualization
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Here I load the image
# I use PIL because it is simple and reliable
# Then I convert it to RGB to make sure the format is always consistent
# Finally I convert it to a NumPy array so I can work with pixel values
def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


# In this step, I convert the image to grayscale
# I do this because working with one channel is much easier than three
# Also, segmentation usually depends on intensity, not color
# I used standard weights because green contributes more to brightness
def to_gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray.astype(np.uint8)


# Here I try to find the best threshold automatically
# The idea is to separate pixels into two groups:
# one for tissue and one for background
# I loop over all possible thresholds and pick the one
# that creates the biggest difference between the two groups
def compute_threshold(gray):
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


# Now I apply the threshold
# If pixel value is lower than threshold, I consider it as tissue
# Otherwise, it is background
# I use 0 for tissue and 255 for background to keep it simple
def apply_threshold(gray, t):
    return np.where(gray <= t, 0, 255).astype(np.uint8)


# This step expands the tissue region
# If any neighbor is tissue, I also consider this pixel as tissue
# This helps fill small gaps and connect broken parts
def dilation(mask):
    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if np.any(mask[i-1:i+2, j-1:j+2] == 0):
                output[i,j] = 0
            else:
                output[i,j] = 255

    return output


# This is the opposite of dilation
# Here I try to remove small noisy pixels
# A pixel stays as tissue only if all its neighbors are tissue
def erosion(mask):
    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if np.all(mask[i-1:i+2, j-1:j+2] == 0):
                output[i,j] = 0
            else:
                output[i,j] = 255

    return output


# Here I combine both operations
# First I expand the regions to fill holes
# Then I shrink them a bit to remove noise
# This simple combination already improves the mask a lot
def refine_segmentation(mask):
    mask = dilation(mask)
    mask = dilation(mask)
    mask = erosion(mask)
    return mask


# Sometimes there are multiple regions after segmentation
# But usually we only care about the main tissue
# So here I find all connected regions and keep only the largest one
def select_main_region(mask):
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


# Here I evaluate how good the segmentation is
# I don’t just check how big the region is
# I also check how consistent it is (coherence)
# If most pixels belong to one region, it means segmentation is clean
def analyze(mask):
    total_pixels = mask.size
    foreground_pixels = np.sum(mask == 0)

    ratio = foreground_pixels / total_pixels

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

    if coherence > 0.9:
        quality = "High"
    elif coherence > 0.6:
        quality = "Medium"
    else:
        quality = "Low"

    return ratio, quality


# Here I create an overlay to visualize the result
# I highlight the boundary of the tissue using a bright color
# This makes it easier to see the segmentation result
def create_overlay(img, mask):
    overlay = img.copy()
    h, w = mask.shape

    for i in range(1, h-1):
        for j in range(1, w-1):
            if mask[i,j] == 0:
                if np.any(mask[i-1:i+2, j-1:j+2] == 255):
                    overlay[i,j] = [255, 0, 0]

    return overlay


# Here I display all results together
# I show each step so I can understand how the image changes
# I also save the final figure so I can reuse it later
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

    # I added this part so the output is saved automatically
    # This is useful for reports or README examples
    import os
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/final_pipeline.png", bbox_inches='tight')

    plt.show()


# This is where everything runs
# I take the input image, process it step by step,
# and finally show and evaluate the result
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

