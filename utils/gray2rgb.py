import numpy as np

from skimage.color import grey2rgb


class Gray2RGB(object):
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
        img = np.swapaxes(np.swapaxes(grey2rgb(img), 1, 2), 0, 1)
        return img
