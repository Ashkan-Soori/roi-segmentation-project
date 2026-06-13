#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import load_image


def test_load_image_returns_valid_structure():
    """
    In this test, I load a real image and check if the result makes sense.

    I am not just checking that the function runs.
    I want to make sure that the output actually looks like a valid image.

    So I verify:
    - it is a NumPy array
    - it has 3 channels (RGB)
    - it has non-zero size
    """

    img = load_image("data/0_1009_0_0_0.jpg")

    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.size > 0

