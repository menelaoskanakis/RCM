from .augmentations import (
    RandomCrop,
    RandomScaling,
    RandomHorizontallyFlip,
    PadImage,
    Compose,
)


def get_composed_transforms(aug_dict):
    """Get a list of augmentations to be performed

    Args:
        aug_dict (dictionary): A dictionary with the different augmentations
            and their corresponding parameters in their own dictionary.
    Returns:
        A list of the augmentations to be performed
    """
    if aug_dict is None:
        return None

    augs = []
    if "RandomScaling" in aug_dict:
        augs.append(RandomScaling(**aug_dict["RandomScaling"]))
    if "PadImage" in aug_dict:
        augs.append(PadImage(**aug_dict["PadImage"]))
    if "RandomCrop" in aug_dict:
        augs.append(RandomCrop(**aug_dict["RandomCrop"]))
    if "RandomHorizontallyFlip" in aug_dict:
        augs.append(RandomHorizontallyFlip(**aug_dict["RandomHorizontallyFlip"]))

    return Compose(augs)
