#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import compute_threshold, apply_threshold


def test_threshold_within_range():
    """
    The threshold should always be within valid intensity bounds.
    """

    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    t = compute_threshold(img)

    assert 0 <= t <= 255


def test_threshold_produces_binary_mask():
    """
    The output mask should contain only two values,
    confirming that the result is a proper binary segmentation.
    """

    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mask = apply_threshold(img, 128)

    unique_vals = np.unique(mask)

    assert set(unique_vals).issubset({0, 255})

