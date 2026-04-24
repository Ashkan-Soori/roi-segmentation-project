#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Simple ROI segmentation tool based on thresholding.

The goal of this script is to separate tissue from background
in breast histology images (H&E and IHC). It uses basic thresholding
followed by simple morphological cleaning to make the mask cleaner.
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    # We first check if the file exists to avoid silent crashes later
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at: {path}")

    image = cv2.imread(path)

    # If OpenCV cannot read it, it usually means the file is corrupted or unsupported
    if image is None:
        raise ValueError("Unable to read the image file.")

    return image


def convert_to_grayscale(image):
    # Thresholding works on intensity values, so we convert to grayscale
    # to simplify the segmentation process
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_otsu_threshold(gray_image):
    # Otsu automatically finds a threshold by analyzing the histogram.
    # This is useful when lighting conditions vary between images.
    threshold_value, mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask, threshold_value


def apply_manual_threshold(gray_image, threshold):
    # In some cases we may want to control the threshold manually,
    # especially if Otsu does not separate tissue well.
    mask = gray_image > threshold
    return (mask * 255).astype("uint8")


def apply_morphology(mask, kernel_size=3):
    # After thresholding, small noise and holes usually appear.
    # Morphological operations help clean the segmentation result.

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Opening removes tiny isolated white pixels
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing fills small black gaps inside the segmented region
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return cleaned


def display_and_save_results(original, gray, raw_mask, cleaned_mask, image_path):
    # Showing all steps together makes it easier to understand
    # how each processing stage affects the image.

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Raw Mask")
    plt.imshow(raw_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("After Morphology")
    plt.imshow(cleaned_mask, cmap="gray")
    plt.axis("off")

    # The Outputs folder is created automatically
    # so the user does not need to create it manually
    output_dir = "Outputs"
    os.makedirs(output_dir, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{image_name}_segmentation_result.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Result saved to: {output_path}")

    plt.show()


def main():
    # Command line arguments allow the user to run the script
    # without modifying the source code.

    parser = argparse.ArgumentParser(
        description="ROI segmentation for H&E and IHC images"
    )

    parser.add_argument("--image", required=True, help="Path to input image")

    parser.add_argument(
        "--method",
        default="otsu",
        choices=["otsu", "manual"],
        help="Thresholding method",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Threshold value if manual method is used",
    )

    parser.add_argument(
        "--kernel",
        type=int,
        default=3,
        help="Kernel size for morphology",
    )

    args = parser.parse_args()

    print("Loading image...")
    image = load_image(args.image)

    print("Converting to grayscale...")
    gray = convert_to_grayscale(image)

    print("Applying threshold...")
    # We allow both automatic and manual thresholding
    # to make the script more flexible.
    if args.method == "otsu":
        raw_mask, t_value = apply_otsu_threshold(gray)
        print(f"Otsu selected threshold: {t_value}")
    else:
        raw_mask = apply_manual_threshold(gray, args.threshold)
        print(f"Manual threshold used: {args.threshold}")

    print("Cleaning mask...")
    cleaned_mask = apply_morphology(raw_mask, args.kernel)

    print("Displaying and saving results...")
    display_and_save_results(image, gray, raw_mask, cleaned_mask, args.image)

    print("Done.")


if __name__ == "__main__":
    main()

