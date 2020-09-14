import os
import torch
import numpy as np
import scipy.io as sio
import zipfile
import sys

from dataloaders.vision import VisionDataset
from torchvision import transforms
from augmentations import get_composed_transforms
from PIL import Image
from collections import OrderedDict
from six.moves import urllib

URL = 'https://data.vision.ee.ethz.ch/kanakism/NYUD_MT.zip'
FILE = 'NYUD_MT.zip'


class NYUDMT(VisionDataset):
    """
    NYU dataset, for multiple tasks
    Included tasks:
        1. Edge detection,
        2. Semantic Segmentation,
        3. Surface Normal prediction,
        5. Depth

        Args:
            root (string): Root directory of the VOC Dataset.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
    """
    def __init__(self,
                 root,
                 image_set='train',
                 edge=False,
                 semseg=False,
                 normals=False,
                 depth=False,
                 transform=None,
                 augmentations=None,
                 download=True):
        super(NYUDMT, self).__init__(root, transform)
        self.root = root
        self.nyud_root = os.path.join(root, 'NYUD_MT')
        self.image_set = image_set

        if (not os.path.isdir(self.nyud_root)) and download:
            self._download()
        elif not os.path.isdir(self.nyud_root):
            raise RuntimeError('Dataset not found or corrupted.')

        # Original Images
        self.im_ids = []
        self.images = []
        image_dir = os.path.join(self.nyud_root, 'images')

        # Task directories
        # Edge Detection
        self.do_edge = edge
        self.edges = []
        _edge_gt_dir = os.path.join(self.nyud_root, 'edge')

        # Semantic segmentation
        self.do_semseg = semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(self.nyud_root, 'segmentation')

        # Surface Normals
        self.do_normals = normals
        self.normals = []
        _normal_gt_dir = os.path.join(self.nyud_root, 'normals')

        # Depth
        self.do_depth = depth
        self.depths = []
        _depth_gt_dir = os.path.join(self.nyud_root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.nyud_root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(self.image_set))

        with open(os.path.join(_splits_dir, self.image_set + '.txt'), 'r') as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(image_dir, line + '.jpg')
            assert os.path.isfile(_image)
            self.images.append(_image)
            self.im_ids.append(line.rstrip('\n'))

            # Edges
            _edge = os.path.join(_edge_gt_dir, line + '.png')
            assert os.path.isfile(_edge)
            self.edges.append(_edge)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_gt_dir, line + '.mat')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            _normal = os.path.join(_normal_gt_dir, line + '.jpg')
            assert os.path.isfile(_normal)
            self.normals.append(_normal)

            _depth = os.path.join(_depth_gt_dir, line + '.mat')
            assert os.path.isfile(_depth)
            self.depths.append(_depth)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))

        # Define augmentations and transformations
        self.augmentations = get_composed_transforms(augmentations)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is the image segmentation.
        """
        sample = OrderedDict()

        _img = self._load_img(index)
        sample['image'] = _img
        sample['labels'] = OrderedDict()
        if self.do_edge:
            _edge = self._load_edge(index)
            assert np.shape(_img)[0:2] == np.shape(_edge)[0:2]
            sample['labels']['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            assert np.shape(_img)[0:2] == np.shape(_semseg)[0:2]
            sample['labels']['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            assert np.shape(_img)[0:2] == np.shape(_normals)[0:2]
            sample['labels']['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            assert np.shape(_img)[0:2] == np.shape(_depth)[0:2]
            sample['labels']['depth'] = _depth

        if self.augmentations is not None:
            sample = self.augmentations(sample)
        sample = self.transformation(sample)

        sample['meta'] = {'image': str(self.im_ids[index]),
                          'im_size': (np.shape(_img)[0], np.shape(_img)[1])}
        return sample

    def __len__(self):
        return len(self.images)

    def transformation(self, sample):
        img = Image.fromarray(sample['image'].astype('uint8'), 'RGB')

        sample['image'] = self.transform(img)
        for key, target in sample['labels'].items():
            assert np.shape(img)[0:2] == np.shape(target)[0:2]
            if key in {'semseg'}:
                sample['labels'][key] = torch.from_numpy(np.array(target)).long()
            elif key in {'depth', 'edge'}:
                sample['labels'][key] = torch.from_numpy(np.array(target)).float()
            elif key in {'normals'}:
                target = np.swapaxes(np.swapaxes(np.array(target), 1, 2), 0, 1)
                sample['labels'][key] = torch.from_numpy(target).float()
        return sample

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_edge(self, index):
        _edge = np.array(Image.open(self.edges[index])).astype(np.float32) / 255.
        return _edge

    def _load_semseg(self, index):
        # Note: Related works are ignoring the background class (40-way classification)
        # Here we conducted using 41-way classification
        _semseg = np.array(sio.loadmat(self.semsegs[index])['segmentation']).astype(np.float32)
        return _semseg

    def _load_normals(self, index):
        _tmp = np.array(Image.open(self.normals[index])).astype(np.float32)
        _normals = 2.0 * _tmp / 255.0 - 1.0
        return _normals

    def _load_depth(self, index):
        _depth = np.array(sio.loadmat(self.depths[index])['depth']).astype(np.float32)
        return _depth

    def _download(self):
        print('Downloading ' + URL + ' to ' + self.root)
        _fpath = os.path.join(self.root, FILE)

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> %s %.1f%%' %
                             (_fpath, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(URL, _fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting zip file')
        with zipfile.ZipFile(_fpath, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        cmd = 'rm {}/{}'.format(self.root, FILE)
        os.system(cmd)
        if os.path.isdir('{}/__MACOSX'.format(self.root)):
            cmd = 'rm -r {}/__MACOSX'.format(self.root)
            os.system(cmd)
        print('Done!')
