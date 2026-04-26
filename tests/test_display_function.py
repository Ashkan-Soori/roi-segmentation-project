#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from unittest.mock import patch
import roi_segmentation.main as main_module


def test_display_function_runs_without_real_plotting():
    """
    This test verifies that the display_and_save_results
    function runs without crashing.
    Plotting and saving are mocked.
    """

    dummy_img = np.ones((20, 20, 3), dtype=np.uint8)
    dummy_gray = np.ones((20, 20), dtype=np.uint8)
    dummy_mask = np.ones((20, 20), dtype=np.uint8)

    with patch("matplotlib.pyplot.show"), \
         patch("matplotlib.pyplot.savefig"), \
         patch("os.makedirs"):

        main_module.display_and_save_results(
            dummy_img,
            dummy_gray,
            dummy_mask,
            dummy_mask,
            "fake.jpg"
        )

