#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import (
    to_gray,
    apply_threshold,
    refine_segmentation,
    fill_holes,
    select_main_region
)


def test_pipeline_produces_valid_roi():
    """
    This is a full pipeline test using synthetic data.

    I simulate an image with a dark region in the center.

    The pipeline should:
    - detect that region
    - keep it as ROI
    - not return an empty or full mask

    This checks real behavior, not just execution.
    """

    img = np.full((50,50,3), 255, dtype=np.uint8)

    # add artificial "tissue"
    img[20:30,20:30] = 10

    gray = to_gray(img)

    mask = apply_threshold(gray, 100)

    refined = refine_segmentation(mask)

    refined = fill_holes(refined)

    final = select_main_region(refined)

    ratio = np.sum(final == 0) / final.size

    # ROI must exist but not dominate entire image
    assert 0 < ratio < 1

