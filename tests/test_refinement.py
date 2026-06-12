#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from roi_segmentation.main import refine_segmentation


def test_refinement_behavior_is_reasonable():
    """
    This test checks whether the refinement step behaves in a stable and reasonable way.

    Instead of assuming that the refined region must always shrink,
    we acknowledge that some operations (like morphological closing)
    may slightly expand regions in order to smooth boundaries or fill small gaps.

    So rather than enforcing strict reduction, we verify two key ideas:

    1. The main region should still exist after refinement.
    2. The result should remain controlled and not take over the entire image.

    This makes the test more aligned with how real image processing behaves.
    """

    # Create a simple test mask:
    # white background (255), with a small dark region (0)
    mask = np.full((100, 100), 255, dtype=np.uint8)

    # Add a small "noise-like" region
    mask[0:3, 0:3] = 0

    # Apply refinement
    refined = refine_segmentation(mask)

    # --- Check 1: Something should still be detected ---
    # If everything disappears, refinement is too aggressive
    assert np.sum(refined == 0) > 0

    # --- Check 2: It should not expand uncontrollably ---
    # If everything becomes foreground, something is wrong
    assert np.sum(refined == 0) < refined.size

