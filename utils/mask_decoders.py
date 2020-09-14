import numpy as np


class VOCSegmentationMaskDecoder(object):
    """Decode predictions to semantic image
    """
    def __init__(self, n_classes):
        self.labels = np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )[0:n_classes]

    def __call__(self, label_mask):
        """
        Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
        Returns:
            rgb (np.ndarray): the resulting decoded color image.
        """
        # Initialize the segmentation map using the "void" colour
        r = np.ones_like(label_mask).astype(np.uint8) * 224
        g = np.ones_like(label_mask).astype(np.uint8) * 223
        b = np.ones_like(label_mask).astype(np.uint8) * 192

        for ind, label_colour in enumerate(self.labels):
            r[label_mask == ind] = label_colour[0]
            g[label_mask == ind] = label_colour[1]
            b[label_mask == ind] = label_colour[2]

        rgb = np.stack([r, g, b], axis=0) / 255.0
        return rgb


class NYUDSegmentationMaskDecoder(object):
    """Decode predictions to semantic image
    """
    def __init__(self, n_classes, has_bg=True):
        n_classes = n_classes + int(has_bg)
        self.labels = self.colormap(n_classes)

    def colormap(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'

        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap

    def __call__(self, label_mask):
        """
        Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
        Returns:
            rgb (np.ndarray): the resulting decoded color image.
        """
        # Initialize the segmentation map using the "void" colour
        r = np.ones_like(label_mask).astype(np.uint8) * 224
        g = np.ones_like(label_mask).astype(np.uint8) * 223
        b = np.ones_like(label_mask).astype(np.uint8) * 192

        for ind, label_colour in enumerate(self.labels):
            r[label_mask == ind] = label_colour[0]
            g[label_mask == ind] = label_colour[1]
            b[label_mask == ind] = label_colour[2]

        rgb = np.stack([r, g, b], axis=0) / 255.0
        return rgb
