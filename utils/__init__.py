from utils.mask_decoders import VOCSegmentationMaskDecoder, NYUDSegmentationMaskDecoder
from utils.denormalize_image import DenormalizeImage
from utils.timer import Timer
from utils.running_performance import RunningMeter
from utils.img_logging import ImgLogging
from utils.prediction_decoding import PredictionDecoder
from utils.gray2rgb import Gray2RGB
from utils.normal2img import Normal2Img


def get_timer(msg):
    """Timer to keep track of training times

    Args:
        msg (str): Message to use at the end of training
    Returns:
        Timer object
    """
    return Timer(msg)


def get_running_meter():
    """Keep track of a value

    Returns:
        Meter object
    """
    return RunningMeter()


def get_img_logging(dataset, tasks):
    """
    """
    decoders_dic = {'image': DenormalizeImage()}
    if dataset == 'PascalContextMT':
        if 'edge' in tasks:
            decoders_dic['edge'] = Gray2RGB()
        if 'human_parts' in tasks:
            decoders_dic['human_parts'] = VOCSegmentationMaskDecoder(7)
        if 'semseg' in tasks:
            decoders_dic['semseg'] = VOCSegmentationMaskDecoder(21)
        if 'normals' in tasks:
            decoders_dic['normals'] = Normal2Img()
        if 'sal' in tasks:
            decoders_dic['sal'] = Gray2RGB()
        return ImgLogging(decoders_dic)
    else:
        return None


def get_pred_decoder(dataset, tasks):
    """
    """
    decoders_dic = {}
    if dataset == 'PascalContextMT':
        if 'edge' in tasks:
            decoders_dic['edge'] = None
        if 'human_parts' in tasks:
            decoders_dic['human_parts'] = VOCSegmentationMaskDecoder(7)
        if 'semseg' in tasks:
            decoders_dic['semseg'] = VOCSegmentationMaskDecoder(21)
        if 'normals' in tasks:
            decoders_dic['normals'] = Normal2Img()
        if 'sal' in tasks:
            decoders_dic['sal'] = None
    elif dataset == 'NYUDMT':
        if 'edge' in tasks:
            decoders_dic['edge'] = None
        if 'semseg' in tasks:
            decoders_dic['semseg'] = NYUDSegmentationMaskDecoder(40)
        if 'normals' in tasks:
            decoders_dic['normals'] = Normal2Img()
        if 'depth' in tasks:
            decoders_dic['depth'] = None
    else:
        raise NotImplementedError("Dataset {} does not include visualization".format(dataset))

    return PredictionDecoder(decoders_dic)
