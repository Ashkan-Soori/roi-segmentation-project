#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from unittest.mock import patch
import numpy as np
from roi_segmentation import main as main_module


@patch("roi_segmentation.main.display_and_save_results")
@patch("roi_segmentation.main.apply_morphology")
@patch("roi_segmentation.main.apply_otsu_threshold")
@patch("roi_segmentation.main.convert_to_grayscale")
@patch("roi_segmentation.main.load_image")
@patch("argparse.ArgumentParser.parse_args")
def test_main_otsu_flow(
    mock_args,
    mock_load,
    mock_gray,
    mock_otsu,
    mock_morph,
    mock_display,
):

    mock_args.return_value.image = "fake.jpg"
    mock_args.return_value.method = "otsu"
    mock_args.return_value.threshold = 200
    mock_args.return_value.kernel = 3

    fake_img = np.zeros((10, 10, 3))
    fake_gray = np.zeros((10, 10))
    fake_mask = np.zeros((10, 10))

    mock_load.return_value = fake_img
    mock_gray.return_value = fake_gray
    mock_otsu.return_value = (fake_mask, 123)
    mock_morph.return_value = fake_mask

    main_module.main()

    mock_load.assert_called_once()
    mock_display.assert_called_once()

