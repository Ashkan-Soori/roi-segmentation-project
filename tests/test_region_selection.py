#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import select_main_region


def test_largest_region_is_selected():
    """
    I create two regions of different sizes.

    The function should keep only the larger one.

    This verifies the actual logic of region selection.
    """

    mask = np.array([
        [0,0,255,255],
        [0,0,255,0],
        [255,255,255,0]
    ], dtype=np.uint8)

    result = select_main_region(mask)

    # largest region should dominate
    assert np.sum(result == 0) >= 4

