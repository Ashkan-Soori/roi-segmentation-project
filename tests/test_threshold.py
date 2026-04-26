#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import apply_manual_threshold, apply_otsu_threshold


def test_manual_threshold_binary_output():
    gray = np.array([[10, 200], [250, 100]], dtype=np.uint8)

    mask = apply_manual_threshold(gray, 150)

    unique_vals = np.unique(mask)
    assert set(unique_vals).issubset({0, 255})


def test_manual_threshold_shape():
    gray = np.zeros((5, 5), dtype=np.uint8)
    mask = apply_manual_threshold(gray, 100)

    assert mask.shape == gray.shape


def test_otsu_threshold_returns_mask_and_value():
    gray = np.zeros((10, 10), dtype=np.uint8)
    mask, t_value = apply_otsu_threshold(gray)

    assert mask.shape == gray.shape
    assert isinstance(t_value, (int, float))

