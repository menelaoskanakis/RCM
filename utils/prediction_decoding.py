import scipy.misc
import os


class PredictionDecoder(object):
    """Maintain track and merge images and their predictions for visualization

    Args:
        decoders_dic (dictionary): Dictionary for decoding all outputs for plotting
    """
    def __init__(self, decoders_dic):
        # Decoders for the different tasks
        self.decoders = decoders_dic

    def decode_pred(self, key, pred):
        if key in self.decoders:
            if self.decoders[key] is not None:
                return self.decoders[key](pred)
            else:
                return pred
        else:
            raise ValueError('Key {} for input decoding is not supported'.format(key))

    def save_pred(self, save_dir, img_name, key, pred):
        decoded_pred = self.decode_pred(key, pred)
        if key in {'human_parts', 'semseg'}:
            scipy.misc.toimage(pred, cmin=0, cmax=255).save(os.path.join(save_dir, key, img_name + '_encoded.png'))
            scipy.misc.toimage(decoded_pred, cmin=0, cmax=1).save(os.path.join(save_dir, key, img_name + '_decoded.png'))
        elif key in {'sal', 'edge', 'normals'}:
            scipy.misc.toimage(decoded_pred, cmin=0, cmax=1).save(os.path.join(save_dir, key, img_name + '.png'))
        elif key in {'depth'}:
            scipy.io.savemat(os.path.join(save_dir, key, img_name + '.mat'), {'depth': decoded_pred})
        else:
            raise ValueError('Key {} for input decoding is not supported'.format(key))
