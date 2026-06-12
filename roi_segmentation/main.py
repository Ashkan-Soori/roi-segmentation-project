#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Simple logging
# -----------------------------
def log(message):
    """
    I didn't want to use a full logging system here because it would be overkill.
    This small helper just makes the output easier to read when the pipeline runs.
    """
    print(f"[INFO] {message}")


# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    """
    First thing I do is check if the file actually exists.
    It's a small thing, but it prevents confusing errors later.

    Then I load the image using OpenCV and convert it to RGB,
    because OpenCV reads images in BGR by default, which is not ideal for visualization.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError("The image could not be loaded. Something is wrong with the file.")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------
# Convert to grayscale
# -----------------------------
def to_gray(img):
    """
    I convert the image to grayscale because thresholding works better on a single channel.
    It also simplifies the problem quite a bit.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# -----------------------------
# Custom Otsu implementation
# -----------------------------
def custom_otsu(gray):
    """
    Instead of using cv2.THRESH_OTSU directly,
    I implemented a simple version myself.

    The idea is to go through all possible thresholds and find the one
    that best separates foreground and background based on variance.

    This is definitely not the most optimized version,
    but it makes the logic much clearer.
    """

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_pixels = gray.size

    best_threshold = 0
    max_variance = 0

    total_sum = np.dot(np.arange(256), hist)

    background_sum = 0
    background_weight = 0

    for t in range(256):
        background_weight += hist[t]

        if background_weight == 0:
            continue

        foreground_weight = total_pixels - background_weight
        if foreground_weight == 0:
            break

        background_sum += t * hist[t]

        mean_bg = background_sum / background_weight
        mean_fg = (total_sum - background_sum) / foreground_weight

        variance = background_weight * foreground_weight * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    _, mask = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)

    return best_threshold, mask


# -----------------------------
# Decide threshold method
# -----------------------------
def choose_method(gray):
    """
    I didn't want to blindly apply Otsu every time.

    So here I use a very simple idea:
    - If the image has low contrast, Otsu might not work well
    - So I fallback to a fixed threshold

    It's not perfect, but it adds a bit of reasoning to the pipeline.
    """

    std = np.std(gray)

    if std < 25:
        log("Image seems low contrast → using manual threshold")
        return "manual"
    else:
        log("Image has enough contrast → using Otsu")
        return "otsu"


# -----------------------------
# Clean mask
# -----------------------------
def clean_mask(mask, kernel_size=3):
    """
    The raw mask is usually messy.

    First I remove small noisy spots using opening,
    then I fill small holes using closing.

    This step makes a big difference in practice.
    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return cleaned


# -----------------------------
# Extract main region
# -----------------------------
def extract_largest(mask):
    """
    After cleaning, there might still be multiple regions.

    I assume that the main tissue is the largest one,
    so I keep only the biggest connected component.

    This is a simple assumption, but it works quite well for this task.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    largest_label = 1
    largest_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > largest_area:
            largest_area = area
            largest_label = i

    final_mask = np.zeros_like(mask)
    final_mask[labels == largest_label] = 255

    return final_mask, largest_area


# -----------------------------
# Analyze result
# -----------------------------
def analyze_mask(mask):
    """
    I wanted to have a simple way to check if the segmentation makes sense.

    So I calculate how much of the image is classified as ROI.
    If it's too small, something probably went wrong.
    """

    total_pixels = mask.size
    roi_pixels = np.sum(mask == 255)

    ratio = roi_pixels / total_pixels

    log(f"ROI ratio: {ratio:.2f}")

    if ratio < 0.05:
        print("[WARNING] The detected region is very small → result might not be reliable")

    return ratio


# -----------------------------
# Show results
# -----------------------------
def show_results(img, gray, raw, clean, final):
    """
    I like to visualize each step of the pipeline.
    It makes debugging much easier and also helps understand what is happening.
    """

    plt.figure(figsize=(16, 4))

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

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    """
    This is where everything comes together.

    The idea is to keep the pipeline simple but structured:
    load → process → clean → analyze → save
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
        log("Running custom Otsu...")
        t, raw = custom_otsu(gray)
    else:
        t = 180
        log("Using manual threshold")
        _, raw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    log(f"Threshold used: {t}")

    log("Cleaning mask...")
    clean = clean_mask(raw, args.kernel)

    log("Extracting main region...")
    final, area = extract_largest(clean)

    log(f"ROI area: {area} pixels")

    ratio = analyze_mask(final)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "final_mask.png")

    cv2.imwrite(output_path, final)
    log(f"Saved result to {output_path}")

    # Save a small report
    with open(os.path.join(args.output, "report.txt"), "w") as f:
        f.write(f"Threshold: {t}\n")
        f.write(f"Area: {area}\n")
        f.write(f"Ratio: {ratio:.2f}\n")

    if args.show:
        show_results(img, gray, raw, clean, final)

    log("Done.")


if __name__ == "__main__":
    main()


# In[ ]:




