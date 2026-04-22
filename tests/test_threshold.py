#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import apply_manual_threshold


def test_manual_threshold_output_values():
    """
    This test checks that the manual threshold function
    produces a binary mask containing only 0 and 255.
    """

    image = np.array([[10, 200], [250, 100]], dtype=np.uint8)

    mask = apply_manual_threshold(image, 150)

    unique_values = np.unique(mask)

    assert set(unique_values).issubset({0, 255})


def test_manual_threshold_shape():
    """
    This test ensures that the output mask
    has the same shape as the input image.
    """

    image = np.zeros((5, 5), dtype=np.uint8)

    mask = apply_manual_threshold(image, 100)

    assert mask.shape == image.shape

