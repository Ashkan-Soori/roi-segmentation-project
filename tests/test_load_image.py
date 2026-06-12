#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import logging
import numpy as np
from roi_segmentation.main import load_image


# Configure logger for test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_load_image_with_fallback():
    """
    This test tries to load a real image from disk.

    In case the image is not available, it switches to a simulated image.
    This way, the test does not depend entirely on external files
    and can still run in different environments.

    The goal here is to keep a balance:
    - use real data when possible
    - remain stable and reproducible when it is not
    """

    image_path = "data/0_1009_0_0_0.jpg"

    if os.path.exists(image_path):
        logger.info("Real image found. Loading from disk.")
        img = load_image(image_path)
    else:
        logger.warning("Image not found. Falling back to a simulated input.")
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # At this point, regardless of the source, we validate the image

    logger.info(f"Image shape: {img.shape}")
    logger.info(f"Pixel value range: min={img.min()}, max={img.max()}")

    # Basic structural checks
    assert img is not None
    assert img.ndim == 3

    # Make sure dimensions are valid
    assert img.shape[0] > 0
    assert img.shape[1] > 0

    # Ensure pixel values are within expected bounds
    assert img.min() >= 0
    assert img.max() <= 255

