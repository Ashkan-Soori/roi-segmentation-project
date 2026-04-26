#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pytest
from unittest.mock import patch
from roi_segmentation.main import load_image


def test_load_image_file_not_exists():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_image("fake.jpg")


def test_load_image_success():
    fake_img = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch("os.path.exists", return_value=True), \
         patch("cv2.imread", return_value=fake_img):

        result = load_image("fake.jpg")
        assert result.shape == (10, 10, 3)

