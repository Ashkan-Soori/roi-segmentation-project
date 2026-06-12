#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import select_main_region


def test_selects_most_significant_region():
    """
    The algorithm should pick the most meaningful region,
    not just any detected component.
    """

    mask = np.full((100, 100), 255, dtype=np.uint8)

    mask[10:20, 10:20] = 0  # small region
    mask[40:90, 40:90] = 0  # large region

    final, _ = select_main_region(mask)

    assert np.sum(final == 0) > 1000

