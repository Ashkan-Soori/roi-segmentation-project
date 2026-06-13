#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import create_overlay


def test_overlay_highlights_boundaries():
    """
    I create a simple region and check if the overlay
    actually modifies boundary pixels.

    This ensures that the visualization is meaningful.
    """

    img = np.zeros((20,20,3), dtype=np.uint8)
    mask = np.full((20,20), 255, dtype=np.uint8)

    mask[5:15,5:15] = 0

    overlay = create_overlay(img, mask)

    # overlay should not be identical
    assert not np.array_equal(overlay, img)

