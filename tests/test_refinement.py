#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import refine_segmentation, fill_holes


def test_refinement_preserves_structure():
    """
    Refinement is allowed to expand regions slightly,
    but it should not remove the region or behave randomly.

    I only check that the region still exists after refinement.
    """

    mask = np.full((50,50), 255, dtype=np.uint8)
    mask[25,25] = 0

    refined = refine_segmentation(mask)

    assert np.sum(refined == 0) > 0


def test_fill_holes_actually_fills_internal_gaps():
    """
    This test creates a proper enclosed hole.

    The white pixel in the center is completely surrounded by tissue,
    so it should be converted to foreground.
    """

    mask = np.array([
        [255,255,255,255,255],
        [255,0,0,0,255],
        [255,0,255,0,255],  # hole
        [255,0,0,0,255],
        [255,255,255,255,255]
    ], dtype=np.uint8)

    result = fill_holes(mask)

    assert result[2,2] == 0

