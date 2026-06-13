#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import analyze


def test_analysis_reflects_coherence():
    """
    I create a fully connected region.

    Since everything is one region, coherence should be high,
    and quality should be labeled as High.
    """

    mask = np.zeros((20,20), dtype=np.uint8)

    ratio, quality = analyze(mask)

    assert ratio == 1.0
    assert quality == "High"


def test_analysis_detects_fragmentation():
    """
    I create two separate regions.

    This should reduce coherence, so quality should not be High.
    """

    mask = np.full((20,20), 255, dtype=np.uint8)

    mask[0:5,0:5] = 0
    mask[15:19,15:19] = 0

    ratio, quality = analyze(mask)

    assert quality in ["Medium", "Low"]

