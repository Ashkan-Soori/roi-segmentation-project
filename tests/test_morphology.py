#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import apply_morphology


def test_morphology_preserves_shape():
    mask = np.zeros((10, 10), dtype=np.uint8)
    cleaned = apply_morphology(mask, kernel_size=3)

    assert cleaned.shape == mask.shape

