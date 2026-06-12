#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import analyze


def test_analysis_returns_valid_values():
    """
    The analysis output should be within expected bounds
    and return a valid label.
    """

    mask = np.full((100, 100), 255, dtype=np.uint8)
    mask[20:80, 20:80] = 0

    ratio, quality = analyze(mask)

    assert 0 <= ratio <= 1
    assert quality in ["Low", "Medium", "High"]

