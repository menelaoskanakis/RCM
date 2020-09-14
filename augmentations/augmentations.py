import numbers
import random
import numpy as np
import cv2


valid_keys = {'image',
              'edge',
              'human_parts',
              'semseg',
              'normals',
              'sal',
              'depth'
              }


class Compose(object):
    """Compose augmentations

    Args:
        augmentations (list of augmentation classes): A list of the augmentations to be performed
    Returns:
        sample: The sample augmented
    """
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, sample):
        for a in self.augmentations:
            img = a(sample)

        return sample


class RandomScaling(object):
    """Random scale the input.
    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.
    Returns:
        sample: The input sample scaled
    """
    def __init__(self, min_scale_factor=1.0, max_scale_factor=1.0, step_size=0):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.step_size = step_size

    def get_random_scale(self, min_scale_factor, max_scale_factor, step_size):
        """Gets a random scaling value.
        Args:
            min_scale_factor: Minimum scale value.
            max_scale_factor: Maximum scale value.
            step_size: The step size from minimum to maximum value.
        Returns:
            A random scale value selected between minimum and maximum value.
        """
        if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
            raise ValueError('Unexpected value of min_scale_factor')

        if min_scale_factor == max_scale_factor:
            min_scale_factor = float(min_scale_factor)
            return min_scale_factor

        # Uniformly sampling of the value from [min, max) when step_size = 0
        if step_size == 0:
            return np.random.uniform(low=min_scale_factor, high=max_scale_factor)
        # Else, randomly select one discrete value from [min, max]
        else:
            num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
            rand_step = np.random.randint(num_steps)
            rand_scale = min_scale_factor + rand_step * step_size
            return rand_scale

    def scale(self, key, unscaled, scale=1.0):
        """Randomly scales image and label.
        Args:
            key: Key indicating the uscaled input origin
            unscaled: Image or target to be scaled.
            scale: The value to scale image and label.
        Returns:
            scaled: The scaled image or target
        """
        # No random scaling if scale == 1.
        if scale == 1.0:
            return unscaled
        image_shape = np.shape(unscaled)[0:2]
        new_dim = tuple([int(x * scale) for x in image_shape])
        if key in {'image', 'normals', 'depth'}:
            scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=cv2.INTER_LINEAR)
        elif key in {'semseg', 'human_parts', 'edge', 'sal'}:
            scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError('Key {} for input origin is not supported'.format(key))

        # we adjust depth maps with rescaling
        if key in {'depth'}:
            scaled /= scale

        return scaled

    def __call__(self, sample):
        img = sample['image']
        random_scale = self.get_random_scale(self.min_scale_factor,
                                             self.max_scale_factor,
                                             self.step_size)
        sample['image'] = self.scale('image', img, scale=random_scale)

        for key, target in sample['labels'].items():
            assert np.shape(img)[0:2] == np.shape(target)[0:2]
            sample['labels'][key] = self.scale(key, target, scale=random_scale)
        return sample


class PadImage(object):
    """Pad image and label to have dimensions >= [size_height, size_width]
    Args:
        size: Desired size
    Returns:
        sample: The input sample padded
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = tuple([int(size), int(size)])
        elif isinstance(size, list):
            self.size = tuple(size)
        else:
            raise ValueError('Crop size must be a number or a list of numbers')

        self.fill_index = {'image': [123, 116, 103],
                           'edge': 255,
                           'human_parts': 255,
                           'semseg': 255,
                           'normals': [0., 0., 0.],
                           'sal': 255,
                           'depth': 0
                           }

    def pad(self, key, unpadded):
        unpadded_shape = np.shape(unpadded)
        delta_height = max(self.size[0] - unpadded_shape[0], 0)
        delta_width = max(self.size[1] - unpadded_shape[1], 0)

        # Location to place image
        height_location = [delta_height // 2, (delta_height // 2) + unpadded_shape[0]]
        width_location = [delta_width // 2, (delta_width // 2) + unpadded_shape[1]]

        pad_value = self.fill_index[key]
        max_height = max(self.size[0], unpadded_shape[0])
        max_width = max(self.size[1], unpadded_shape[1])
        if key in {'image', 'normals'}:
            padded = np.ones((max_height, max_width, 3)) * pad_value
            padded[height_location[0]:height_location[1], width_location[0]:width_location[1], :] = unpadded
        elif key in {'semseg', 'human_parts', 'edge', 'sal', 'depth'}:
            padded = np.ones((max_height, max_width)) * pad_value
            padded[height_location[0]:height_location[1], width_location[0]:width_location[1]] = unpadded
        else:
            raise ValueError('Key {} for input origin is not supported'.format(key))

        return padded

    def __call__(self, sample):
        img = sample['image']
        sample['image'] = self.pad('image', img)

        for key, target in sample['labels'].items():
            assert np.shape(img)[0:2] == np.shape(target)[0:2]
            sample['labels'][key] = self.pad(key, target)
        return sample


class RandomCrop(object):
    """Random crop image if it exceeds desired size
    Args:
        size: Desired size
    Returns:
        sample: The input sample randomly cropped
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = tuple([int(size), int(size)])
        elif isinstance(size, list):
            self.size = tuple(size)
        else:
            raise ValueError('Crop size must be a number or a list of numbers')

        self.padding = PadImage(size)

    def get_random_crop_loc(self, key, uncropped):
        """Gets a random crop location.
        Args:
            key: Key indicating the uncropped input origin
            uncropped: Image or target to be cropped.
        Returns:
            Cropping region.
        """
        uncropped_shape = np.shape(uncropped)
        img_height = uncropped_shape[0]
        img_width = uncropped_shape[1]

        desired_height = self.size[0]
        desired_width = self.size[1]
        if img_height == desired_height and img_width == desired_width:
            return None
        else:
            # Get random offset uniformly from [0, max_offset)
            max_offset_height = img_height - desired_height
            max_offset_width = img_width - desired_width

            offset_height = random.randint(0, max_offset_height)
            offset_width = random.randint(0, max_offset_width)
            crop_loc = {'height': [offset_height, offset_height + desired_height],
                        'width': [offset_width, offset_width + desired_width],
                        }
            return crop_loc

    def random_crop(self, key, uncropped, crop_loc):
        if not crop_loc:
            return uncropped
        else:
            if key in {'image', 'normals'}:
                cropped = uncropped[crop_loc['height'][0]:crop_loc['height'][1],
                          crop_loc['width'][0]:crop_loc['width'][1], :]
            elif key in {'semseg', 'human_parts', 'edge', 'sal', 'depth'}:
                cropped = uncropped[crop_loc['height'][0]:crop_loc['height'][1],
                          crop_loc['width'][0]:crop_loc['width'][1]]
            else:
                raise ValueError('Key {} for input origin is not supported'.format(key))
            assert np.shape(cropped)[0:2] == self.size
            return cropped

    def __call__(self, sample):
        # Ensure the image is at least as large as the desired size
        sample = self.padding(sample)

        img = sample['image']
        crop_location = self.get_random_crop_loc('image', img)
        sample['image'] = self.random_crop('image', img, crop_location)

        for key, target in sample['labels'].items():
            assert np.shape(img)[0:2] == np.shape(target)[0:2]
            sample['labels'][key] = self.random_crop(key, target, crop_location)
        return sample


class RandomHorizontallyFlip(object):
    """Random horizontal flip
    Args:
        p: Probability of flip
    Returns:
        img, mask: The augmented image and mask
    """
    def __init__(self, p=0.5):
        self.p = p

    def flip(self, key, target):
        new_target = np.zeros_like(target)
        if key in {'image', 'normals'}:
            for i in range(np.shape(target)[1]):
                new_target[:, i, :] = target[:, -(i+1), :]
        elif key in {'semseg', 'human_parts', 'edge', 'sal', 'depth'}:
            for i in range(np.shape(target)[1]):
                new_target[:, i] = target[:, -(i+1)]
        else:
            raise ValueError('Key {} for input origin is not supported'.format(key))
        assert np.shape(target)[0:2] == np.shape(new_target)[0:2]
        return new_target

    def __call__(self, sample):
        if random.random() < self.p:
            img = sample['image']
            sample['image'] = self.flip('image', img)

            for key, target in sample['labels'].items():
                assert np.shape(img)[0:2] == np.shape(target)[0:2]
                sample['labels'][key] = self.flip(key, target)
                if key == 'normals':
                    sample['labels'][key][:, :, 0] *= -1
        return sample

