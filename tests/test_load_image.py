#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import load_image


def test_load_image_returns_valid_structure():
    """
    In this test, I load a real image from disk and verify that
    the output has the expected structure and properties.

    The goal is not only to check that the function runs,
    but to ensure that the result is a valid image representation.

    Specifically, I check that:
    - the output is a NumPy array (so it can be processed numerically)
    - the image has three dimensions (height, width, channels)
    - it contains exactly three channels (RGB format)
    - it is not empty
    - pixel values are within the valid range [0, 255]

    These checks confirm that the image is correctly loaded
    and ready for further processing in the pipeline.
    """

    img = load_image("data/0_1009_0_0_0.jpg")

    # structure checks
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.size > 0

    # value checks (to ensure it's a valid image)
    assert img.dtype == np.uint8
    assert np.min(img) >= 0
    assert np.max(img) <= 255

    # sanity check: image should not be completely empty or uniform
    assert not np.all(img == 0)

