#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def test_threshold_separates_intensity_groups():
    """
    Instead of forcing an exact range, I check if the threshold
    actually separates dark and bright values correctly.
    """

    gray = np.array([
        [10, 20, 15],
        [200, 210, 220]
    ], dtype=np.uint8)

    t = compute_threshold(gray)

    mask = apply_threshold(gray, t)

    # dark values should become foreground
    assert mask[0,0] == 0

    # bright values should become background
    assert mask[1,0] == 255

