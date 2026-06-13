#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ROI Segmentation Pipeline (Manual Implementation)

In this project, I tried to build a simple image segmentation pipeline
completely from scratch.

I did NOT use OpenCV or any advanced computer vision libraries.
Instead, I used NumPy to manually implement each step.

The main idea is:
Take an image → convert to grayscale → separate tissue from background →
clean the result → keep the main region → visualize everything.

I also added some improvements:
- removing small holes inside the tissue
- making the overlay boundaries more visible
- evaluating quality based on how consistent the region is

The goal is not perfection, but understanding how things actually work.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):
    """
    Here I simply load the image using PIL.

    I convert it to RGB just to make sure the format is always consistent.
    Then I convert it into a NumPy array so I can work with pixel values easily.
    """
    return np.array(Image.open(path).convert("RGB"))


def to_gray(img):
    """
    In this step, I convert the image to grayscale.

    Instead of using a built-in function, I manually combine
    the R, G, and B channels using standard weights.

    These weights are not random — they reflect how humans
    perceive brightness.
    """
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.299*r + 0.587*g + 0.114*b
    return gray.astype(np.uint8)


def compute_threshold(gray):
    """
    Here I try to automatically find a good threshold.

    I use a simple Otsu-like idea:
    I test all possible thresholds and pick the one that best
    separates the image into two groups.

    The goal is to separate tissue from background in a reasonable way.
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

        # this measures how different the two groups are
        var = bg_weight * fg_weight * (mean_bg - mean_fg) ** 2

        if var > max_var:
            max_var = var
            best_t = t

    return best_t


def apply_threshold(gray, t):
    """
    Now I convert the grayscale image into a binary mask.

    My assumption:
    darker pixels = tissue
    brighter pixels = background

    So I turn it into black (0) and white (255).
    """
    return np.where(gray <= t, 0, 255).astype(np.uint8)


def dilation(mask):
    """
    This step expands the black region (tissue).

    If even one neighbor is tissue, I consider the pixel as tissue.

    This helps fill small gaps inside the region.
    """
    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if np.any(mask[i-1:i+2, j-1:j+2] == 0):
                output[i,j] = 0
            else:
                output[i,j] = 255

    return output


def erosion(mask):
    """
    Opposite of dilation.

    Here I shrink the region slightly.

    A pixel stays as tissue ONLY if all neighbors are also tissue.

    This helps remove noise.
    """
    h, w = mask.shape
    output = mask.copy()

    for i in range(1, h-1):
        for j in range(1, w-1):
            if np.all(mask[i-1:i+2, j-1:j+2] == 0):
                output[i,j] = 0
            else:
                output[i,j] = 255

    return output


def refine_segmentation(mask):
    """
    This is where I try to clean the mask.

    First I expand the region (dilation) to fill holes,
    then I shrink it a bit (erosion) to remove extra noise.

    It's a simple trick but works quite well.
    """
    mask = dilation(mask)
    mask = dilation(mask)
    mask = erosion(mask)
    return mask


def fill_holes(mask):
    """
    After refinement, sometimes there are still small white spots
    inside the tissue.

    These are basically holes.

    So I detect those regions and convert them into tissue.

    This makes the final result cleaner.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    inv = np.where(mask == 0, 255, 0)

    def dfs(x, y):
        stack = [(x, y)]

        while stack:
            i, j = stack.pop()

            if i < 0 or i >= h or j < 0 or j >= w:
                continue

            if visited[i,j] or inv[i,j] != 0:
                continue

            visited[i,j] = True

            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    stack.append((i+dx, j+dy))

    # start from borders
    for i in range(h):
        if inv[i,0] == 0: dfs(i,0)
        if inv[i,w-1] == 0: dfs(i,w-1)

    for j in range(w):
        if inv[0,j] == 0: dfs(0,j)
        if inv[h-1,j] == 0: dfs(h-1,j)

    # fill internal holes
    for i in range(h):
        for j in range(w):
            if inv[i,j] == 0 and not visited[i,j]:
                mask[i,j] = 0

    return mask


def select_main_region(mask):
    """
    Sometimes there are multiple regions.

    I only keep the biggest one, assuming it's the main tissue.

    I use DFS to explore connected pixels.
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


def analyze(mask):
    """
    Instead of just checking how big the region is,
    I check how consistent it is.

    If most pixels belong to one connected region,
    I consider the result good.
    """
    total = mask.size
    foreground = np.sum(mask == 0)
    ratio = foreground / total

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
    coherence = largest / foreground

    if coherence > 0.9:
        quality = "High"
    elif coherence > 0.6:
        quality = "Medium"
    else:
        quality = "Low"

    return ratio, quality


def create_overlay(img, mask):
    """
    I highlight the boundary of the tissue.

    Instead of marking just one pixel,
    I make it thicker so it's easier to see.
    """
    overlay = img.copy()
    h, w = mask.shape

    color = np.array([0, 255, 0])

    for i in range(1, h-1):
        for j in range(1, w-1):
            if mask[i,j] == 0:
                if np.any(mask[i-1:i+2, j-1:j+2] == 255):
                    for dx in [-1,0,1]:
                        for dy in [-1,0,1]:
                            overlay[i+dx, j+dy] = color

    return overlay


def show_results(img, gray, mask, refined, final, t, ratio, quality):
    """
    Finally, I show all steps side by side.

    This helps me understand how each step changes the result.
    """
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

    refined = fill_holes(refined)

    final = select_main_region(refined)

    ratio, quality = analyze(final)

    print(f"Threshold: {t}")
    print(f"Ratio: {ratio:.2f}")
    print(f"Quality: {quality}")

    show_results(img, gray, mask, refined, final, t, ratio, quality)

