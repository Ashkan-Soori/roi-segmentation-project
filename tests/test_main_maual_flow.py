#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from unittest.mock import patch
import roi_segmentation.main as main_module


def test_main_manual_flow_executes_full_pipeline():
    """
    This test executes the full main() pipeline using
    manual thresholding. External dependencies like file I/O
    and plotting are mocked so the test focuses only on logic flow.
    """

    dummy_image = np.ones((50, 50, 3), dtype=np.uint8) * 200
    dummy_gray = np.ones((50, 50), dtype=np.uint8) * 200
    dummy_mask = np.ones((50, 50), dtype=np.uint8) * 255

    test_args = [
        "program_name",
        "--image", "fake_path.jpg",
        "--method", "manual",
        "--threshold", "150",
        "--kernel", "5",
    ]

    with patch("sys.argv", test_args), \
         patch.object(main_module, "load_image", return_value=dummy_image), \
         patch.object(main_module, "convert_to_grayscale", return_value=dummy_gray), \
         patch.object(main_module, "apply_manual_threshold", return_value=dummy_mask), \
         patch.object(main_module, "apply_morphology", return_value=dummy_mask), \
         patch.object(main_module, "display_and_save_results") as mock_display:

        main_module.main()

        # We verify that display function was called,
        # meaning the full pipeline executed successfully.
        mock_display.assert_called_once()

