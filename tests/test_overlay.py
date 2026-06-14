#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import create_overlay


def test_overlay_highlights_boundaries():
    img = np.zeros((20,20,3), dtype=np.uint8)
    mask = np.full((20,20), 255, dtype=np.uint8)

    mask[5:15,5:15] = 0

    overlay = create_overlay(img, mask)

    # 🔥 check that something changed
    assert not np.array_equal(overlay, img)

    # 🔥 check that boundary pixels are modified
    # boundary around the square (approx)
    boundary_pixel = overlay[5,5]

    assert not np.array_equal(boundary_pixel, img[5,5])

