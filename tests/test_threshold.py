#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import compute_threshold, apply_threshold


def test_threshold_separates_intensity_groups():
    """
    Here I made a very simple image with two types of pixels:
    some are dark and some are bright.

    I don’t expect the threshold to be a specific number,
    because the algorithm might choose different values.

    What really matters is this:
    after applying the threshold, dark pixels should go to one side
    and bright pixels should go to the other side.

    So basically I am checking if the separation actually works.
    """

    gray = np.array([
        [10, 20, 15],
        [200, 210, 220]
    ], dtype=np.uint8)

    t = compute_threshold(gray)

    mask = apply_threshold(gray, t)

    # this pixel is dark → should become foreground (0)
    assert mask[0, 0] == 0

    # this pixel is bright → should become background (255)
    assert mask[1, 0] == 255


def test_apply_threshold_creates_binary_mask():
    """
    In this test I just want to make sure the output is clean.

    After thresholding, the result should only contain
    two values: 0 and 255.

    If I see anything else, it means something is wrong.
    """

    gray = np.array([
        [10, 200],
        [50, 180]
    ], dtype=np.uint8)

    mask = apply_threshold(gray, 100)

    unique_values = np.unique(mask)

    assert set(unique_values).issubset({0, 255})

