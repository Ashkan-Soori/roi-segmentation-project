#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ROI Segmentation Pipeline (Manual + Coherence-Based Evaluation)

In this project, I tried to build a full image segmentation pipeline from scratch.

Instead of using ready-made functions from libraries like OpenCV,
I implemented everything manually so I could really understand what is happening
in each step.

The main idea is to take a medical image and extract the region of interest (ROI),
which in this case is the tissue area.

The pipeline is:
image → grayscale → threshold → clean → main region → hole filling → analysis → visualization
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Here I load the image from disk.
# I use PIL because it is simple and works well with most image formats.
# I convert the image to RGB to make sure everything is consistent.
# Then I convert it to a NumPy array because I want to work directly with pixel values.
def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


# In this step, I convert the image into grayscale.
# The reason is that segmentation usually depends on intensity, not color.
# So instead of working with 3 channels, I reduce it to 1 channel.
# I use standard weights because green contributes more to brightness perception.
def to_gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray.astype(np.uint8)


# Here I try to automatically find a good threshold.
# The goal is to separate the image into two parts:
# foreground (tissue) and background.
# I test all possible threshold values and choose the one
# that maximizes the difference between the two groups.
# This is similar to Otsu's method.
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


# After finding the threshold, I convert the grayscale image to a binary mask.
# Pixels darker than the threshold are considered tissue (0).
# Pixels brighter than the threshold are background (255).
# I use 0 and 255 because it is easier to visualize later.
def apply_threshold(gray, t):
    return np.where(gray <= t, 0, 255).astype(np.uint8)


# Dilation expands the black regions (tissue).
# If a pixel has at least one neighbor that is tissue,
# I convert it to tissue as well.
# This helps fill small gaps and connect broken parts.
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


# Erosion does the opposite of dilation.
# It removes small noise.
# A pixel remains tissue only if all its neighbors are also tissue.
# This is useful for cleaning the segmentation result.
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


# Here I combine dilation and erosion to improve the mask.
# First I apply dilation multiple times to fill holes and connect regions.
# Then I apply erosion to remove extra noise.
# This makes the segmentation much cleaner.
def refine_segmentation(mask):
    mask = dilation(mask)
    mask = dilation(mask)
    mask = dilation(mask)

    mask = erosion(mask)
    mask = erosion(mask)

    return mask


# Sometimes the segmentation produces multiple separate regions.
# But usually we only care about the main tissue.
# So here I find all connected components and keep only the largest one.
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


# Even after all processing, small white holes may remain inside the tissue.
# These are not real background, they are just segmentation errors.
# So here I fill those holes using a flood-fill idea.
# I start from the borders and mark real background.
# Anything that is not connected to the border is considered a hole and filled.
def fill_holes(mask):
    h, w = mask.shape
    filled = mask.copy()
    visited = np.zeros_like(mask, dtype=bool)

    stack = []

    for i in range(h):
        stack.append((i,0))
        stack.append((i,w-1))
    for j in range(w):
        stack.append((0,j))
        stack.append((h-1,j))

    while stack:
        i, j = stack.pop()

        if i < 0 or i >= h or j < 0 or j >= w:
            continue

        if visited[i,j] or filled[i,j] != 255:
            continue

        visited[i,j] = True

        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                stack.append((i+dx, j+dy))

    for i in range(h):
        for j in range(w):
            if filled[i,j] == 255 and not visited[i,j]:
                filled[i,j] = 0

    return filled


# Here I evaluate how good the segmentation is.
# I check how much of the image is selected (ratio)
# and also how consistent the region is (coherence).
# If most pixels belong to one region, the result is good.
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


# Here I create an overlay image.
# I highlight the boundary of the ROI using red color.
# I also make the boundary thicker so it is easier to see.
def create_overlay(img, mask):
    overlay = img.copy()
    h, w = mask.shape

    thickness = 3

    for i in range(1, h-1):
        for j in range(1, w-1):
            if mask[i,j] == 0:
                if np.any(mask[i-1:i+2, j-1:j+2] == 255):

                    for di in range(-thickness, thickness+1):
                        for dj in range(-thickness, thickness+1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                overlay[ni, nj] = [255, 0, 0]

    return overlay


# Finally, I visualize all steps.
# This helps me understand what happens at each stage.
# I also save the final result so I can use it in reports.
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

    import os
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/final_pipeline.png", bbox_inches='tight')

    plt.show()


# This is the main part where everything runs step by step.
# I take the input image, process it, evaluate it,
# and finally show and save the result.
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

    final = fill_holes(final)

    ratio, quality = analyze(final)

    print(f"Threshold: {t}")
    print(f"Ratio: {ratio:.2f}")
    print(f"Quality: {quality}")

    show_results(img, gray, mask, refined, final, t, ratio, quality)

