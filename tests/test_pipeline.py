#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import (
    apply_threshold,
    refine_segmentation,
    select_main_region
)


def test_pipeline_detects_meaningful_region():
    """
    This test checks the overall behavior of the pipeline.

    The goal is not just to verify the output format,
    but to ensure that a meaningful region is actually detected.
    """

    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    mask = apply_threshold(img, 128)
    clean = refine_segmentation(mask)
    final, _ = select_main_region(clean)

    assert final.shape == img.shape

    # Make sure something is detected
    assert np.sum(final == 0) > 0

    # Make sure it's not the entire image
    assert np.sum(final == 0) < final.size

