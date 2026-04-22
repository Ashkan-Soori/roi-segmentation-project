#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    """
    Load the image from disk.
    We check first if the file exists to avoid unexpected crashes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The image file was not found at: {path}")

    image = cv2.imread(path)
    if image is None:
        raise ValueError("The image could not be opened. Please check the file.")

    # Convert BGR (OpenCV default) to RGB for correct visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def convert_to_grayscale(image_rgb):
    """
    Convert RGB image to grayscale.
    Thresholding works on single-channel images.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return gray


def apply_otsu_threshold(gray_image):
    """
    Apply Otsu thresholding.
    The threshold value is chosen automatically.
    """
    threshold_value, mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold_value, mask


def apply_manual_threshold(gray_image, threshold_value):
    """
    Apply manual thresholding using a user-defined value.
    """
    _, mask = cv2.threshold(
        gray_image, threshold_value, 255, cv2.THRESH_BINARY
    )
    return mask


def apply_morphology(binary_mask, kernel_size=3):
    """
    Clean the binary mask using morphological operations.

    First we remove small noisy regions (opening),
    then we fill small holes inside the ROI (closing).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return cleaned


def display_and_save_results(original, gray, mask_raw, mask_cleaned,
                             method_name):
    """
    Show original image, grayscale, raw mask and cleaned mask.
    Also save the final cleaned result.
    """

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask_raw, cmap="gray")
    plt.title(f"{method_name} (Raw)")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(mask_cleaned, cmap="gray")
    plt.title(f"{method_name} + Morphology")
    plt.axis("off")

    plt.tight_layout()

    output_name = "roi_segmentation_result.png"
    plt.savefig(output_name, dpi=300, bbox_inches="tight")

    print("Result saved as:", output_name)

    plt.show()


def main():
    """
    Main pipeline:
    1. Load image
    2. Convert to grayscale
    3. Apply thresholding (Otsu or manual)
    4. Apply morphology
    5. Display and save results
    """

    parser = argparse.ArgumentParser(
        description="ROI segmentation using thresholding and morphological cleaning."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image."
    )

    parser.add_argument(
        "--method",
        choices=["otsu", "manual"],
        default="otsu",
        help="Thresholding method (default: otsu)."
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=230,
        help="Manual threshold value (used only if method=manual)."
    )

    parser.add_argument(
        "--kernel",
        type=int,
        default=3,
        help="Kernel size for morphological operations (default: 3)."
    )

    args = parser.parse_args()

    try:
        print("Loading image...")
        image = load_image(args.image)

        print("Converting to grayscale...")
        gray = convert_to_grayscale(image)

        if args.method == "otsu":
            print("Applying Otsu threshold...")
            _, mask_raw = apply_otsu_threshold(gray)
            method_name = "Otsu"
        else:
            print("Applying manual threshold...")
            mask_raw = apply_manual_threshold(gray, args.threshold)
            method_name = "Manual"

        print("Cleaning mask using morphology...")
        mask_cleaned = apply_morphology(mask_raw, args.kernel)

        print("Displaying results...")
        display_and_save_results(
            image, gray, mask_raw, mask_cleaned, method_name
        )

        print("Processing completed.")

    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()

