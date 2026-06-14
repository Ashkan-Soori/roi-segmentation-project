#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def test_overlay_highlights_boundaries():
    img = np.zeros((20,20,3), dtype=np.uint8)
    mask = np.full((20,20), 255, dtype=np.uint8)

    mask[5:15,5:15] = 0

    overlay = create_overlay(img, mask)

    # boundary pixel should be red
    assert (overlay[5,10] == [255,0,0]).all()

    # inside region should stay unchanged
    assert (overlay[10,10] == [0,0,0]).all()

    # outside region should stay unchanged
    assert (overlay[0,0] == [0,0,0]).all()

