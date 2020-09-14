import numpy as np


class ImgLogging(object):
    """Maintain track and merge images and their predictions for visualization

    Args:
        decoders_dic (dictionary): Dictionary for decoding all outputs for plotting
    """
    def __init__(self, decoders_dic):
        self.imgs = []
        # Decoders for the different tasks
        self.decoders = decoders_dic

    def decode_img(self, dic):

        img_labels = []
        for ind, (key, label) in enumerate(dic.items()):
            if 'image' in key:
                img_labels = [self.decoders['image'](dic['image'])]
            elif key in self.decoders:
                img_labels.append(self.decoders[key](label['pred']))
                img_labels.append(self.decoders[key](label['gt']))
            else:
                raise ValueError('Key {} for input decoding is not supported'.format(key))

        return img_labels

    def merge(self, img_list, row=True):
        num_imgs, channels, height, width = np.shape(img_list)
        if row:
            total_width = width * num_imgs
            total_height = height
        else:
            total_height = height * num_imgs
            total_width = width

        new_im = np.zeros((3, total_height, total_width))
        if row:
            width_start = 0
            width_end = width
            height_start = 0
            height_end = total_height
        else:
            width_start = 0
            width_end = total_width
            height_start = 0
            height_end = height
        for im in img_list:
            new_im[:, height_start:height_end, width_start:width_end] = im
            if row:
                width_start += width
                width_end = width_start + width
            else:
                height_start += height
                height_end = height_start + height
        return new_im

    def merge_img_labels(self, dic):
        self.imgs.append(self.merge(self.decode_img(dic)))

    def log_imgs(self, writer, iteration):
        writer.add_image('image', self.merge(self.imgs, row=False), iteration)
        self.imgs = []
