import numpy as np


class Normal2Img(object):
    """Convert gray image to rgb representation for plotting

    Args:
        img: Normalized image
    Returns:
        img: Denormalized image
    """
    def __init__(self):
        """
        """

    def __call__(self, img):
        """
        """
        img = (img + 1.0) / 2.0
        return img
