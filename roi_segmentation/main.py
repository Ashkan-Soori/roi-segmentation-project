#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    """
    First thing I do is loading the image from disk.

    I added this check because I ran into issues before where
    the path was wrong and OpenCV just failed silently.
    It's better to stop early and show a clear error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    # Sometimes OpenCV returns None instead of throwing an error
    if img is None:
        raise ValueError("Could not read the image file.")

    # OpenCV reads images in BGR, but matplotlib expects RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(img):
    """
    Convert the image to grayscale.

    I do this because thresholding works much better
    on a single channel instead of RGB.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def custom_otsu(gray):
    """
    Here I tried to implement Otsu thresholding manually.

    Instead of directly calling OpenCV's built-in function,
    I compute the histogram and search for the threshold
    that maximizes the separation between foreground and background.

    This is not the fastest approach, but it makes the logic clearer.
    """
    hist = np.histogram(gray.flatten(), bins=256)[0]
    total = gray.size

    sum_total = sum(i * hist[i] for i in range(256))

    sumB = 0
    wB = 0
    max_var = 0
    threshold = 0

    # I loop over all possible threshold values
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue

        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]

        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        # Between-class variance
        var_between = wB * wF * (mB - mF) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    # Apply the threshold we found
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return threshold, mask


def clean_mask(mask, k=3):
    """
    After thresholding, the mask is usually quite noisy.

    I use two simple morphological operations:
    - Opening: removes small noisy dots
    - Closing: fills small holes inside the region

    The kernel size can be adjusted if needed.
    """
    kernel = np.ones((k, k), np.uint8)

    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    return m


def keep_largest_region(mask):
    """
    In most histology images, the main tissue region
    is the largest connected component.

    So here I extract all contours and keep only the biggest one.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return mask

    largest = max(contours, key=cv2.contourArea)

    new_mask = np.zeros_like(mask)
    cv2.drawContours(new_mask, [largest], -1, 255, thickness=-1)

    return new_mask


def show(img, gray, raw, clean, roi, t):
    """
    This is just for visualization.

    I like to see all intermediate steps side by side
    to understand what each part of the pipeline is doing.
    """
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Gray")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(raw, cmap="gray")
    plt.title(f"Raw (t={t})")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(clean, cmap="gray")
    plt.title("Clean")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(roi, cmap="gray")
    plt.title("Final ROI")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """
    This is the main pipeline of the program.

    The idea is simple:
    1. Load the image
    2. Convert it to grayscale
    3. Apply thresholding (custom Otsu)
    4. Clean the mask
    5. Keep only the main region
    6. Save the result
    7. Optionally display everything
    """

    parser = argparse.ArgumentParser(description="ROI segmentation CLI tool")

    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size")
    parser.add_argument("--output", default="outputs", help="Output folder")

    # This flag is optional: only used if we want to display results
    parser.add_argument("--show", action="store_true", help="Display results")

    args = parser.parse_args()

    print("Loading image...")
    img = load_image(args.image)

    gray = to_gray(img)

    print("Running custom Otsu...")
    t, raw = custom_otsu(gray)

    print("Cleaning mask...")
    clean = clean_mask(raw, args.kernel)

    print("Extracting ROI...")
    roi = keep_largest_region(clean)

    # Just some quick stats to understand the segmentation result
    roi_area = np.sum(roi > 0)
    total_area = roi.size
    roi_percent = (roi_area / total_area) * 100

    print(f"Threshold: {t}")
    print(f"ROI percentage: {roi_percent:.2f}%")

    # Make sure output folder exists
    os.makedirs(args.output, exist_ok=True)

    output_path = os.path.join(args.output, "final_mask.png")

    cv2.imwrite(output_path, roi)

    print(f"Saved to: {output_path}")

    # Show results only if user asked for it
    if args.show:
        show(img, gray, raw, clean, roi, t)


if __name__ == "__main__":
    main()


# In[ ]:




