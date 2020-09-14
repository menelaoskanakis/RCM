import os
import numpy as np
import json
import scipy.io as sio
import cv2
import torch
import sys
import zipfile

from dataloaders.vision import VisionDataset
from augmentations import get_composed_transforms
from PIL import Image
from skimage.morphology import thin
from collections import OrderedDict
from torchvision import transforms
from six.moves import urllib


HUMAN_PART = {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
              'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
              'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
              'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2
              }

URL = 'https://data.vision.ee.ethz.ch/kanakism/PASCAL_MT.zip'
FILE = 'PASCAL_MT.zip'


class PascalContextMT(VisionDataset):
    """
    PASCAL-Context dataset, for multiple tasks
    Included tasks:
        1. Edge detection,
        2. Semantic Segmentation,
        3. Human Part Segmentation,
        4. Surface Normal prediction (distilled),
        5. Saliency (distilled)

        Args:
            root (string): Root directory of the VOC Dataset.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        """
    def __init__(self,
                 root,
                 image_set='train',
                 edge=False,
                 human_parts=False,
                 semseg=False,
                 normals=False,
                 sal=False,
                 transform=None,
                 augmentations=None,
                 download=True):
        super(PascalContextMT, self).__init__(root, transform)
        self.root = root
        self.voc_root = os.path.join(root, 'PASCAL_MT')

        if (not os.path.isdir(self.voc_root)) and download:
            self._download()
        elif not os.path.isdir(self.voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        image_dir = os.path.join(self.voc_root, 'JPEGImages')

        # Task directories
        # Edge Detection
        self.do_edge = edge
        # Context dir: Used for Edge Detection and Normals
        context_dir = os.path.join(self.voc_root, 'Context', 'trainval')

        # Human Part Segmentation
        self.do_human_parts = human_parts
        part_dir = os.path.join(self.voc_root, 'PascalParts')
        self.cat_part = json.load(open(os.path.join(self.voc_root, 'json/pascal_part.json'), 'r'))
        self.human_parts_category = 15
        self.cat_part[str(self.human_parts_category)] = HUMAN_PART
        self.parts_file = os.path.join(self.voc_root, 'ImageSets/Parts', image_set + '.txt')

        # Semantic Segmentation
        self.do_semseg = semseg
        semseg_dir = os.path.join(self.voc_root, 'SemanticSegmentation', 'Context')

        # Normal detection
        self.do_normals = normals
        normal_dir = os.path.join(self.voc_root, 'NormalsDistill')
        if self.do_normals:
            with open(os.path.join(self.voc_root, 'json/nyu_classes.json')) as f:
                cls_nyu = json.load(f)
            with open(os.path.join(self.voc_root, 'json/context_classes.json')) as f:
                cls_context = json.load(f)
            # Find common classes between the two datasets to use for normals
            self.normals_valid_classes = []
            for cl_nyu in cls_nyu:
                if cl_nyu in cls_context and cl_nyu != 'unknown':
                    self.normals_valid_classes.append(cls_context[cl_nyu])
            # Custom additions due to incompatibilities
            self.normals_valid_classes.append(cls_context['tvmonitor'])

        # Saliency detection
        self.do_sal = sal
        sal_dir = os.path.join(self.voc_root, 'SaliencyDistill')

        # Define augmentations and transformations
        self.augmentations = get_composed_transforms(augmentations)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        splits_dir = os.path.join(self.voc_root, 'ImageSets/Context')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered! Please select an appropriate one.')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        print("Initializing dataloader for PASCAL Context {} set".format(image_set))
        self.im_ids = file_names
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.context = [os.path.join(context_dir, x + ".mat") for x in file_names]
        self.parts = [os.path.join(part_dir, x + ".mat") for x in file_names]
        self.semsegs = [os.path.join(semseg_dir, x + ".png") for x in file_names]
        self.normals = [os.path.join(normal_dir, x + ".png") for x in file_names]
        self.sals = [os.path.join(sal_dir, x + ".png") for x in file_names]

        assert (len(self.images) == len(self.context))
        assert (len(self.images) == len(self.parts))
        assert (len(self.images) == len(self.semsegs))
        assert (len(self.images) == len(self.normals))
        assert (len(self.images) == len(self.sals))

        if not self._check_preprocess_parts():
            print('Pre-processing PASCAL dataset for human parts, this will take long, but will be done only once.')
            self._preprocess_parts()

        if self.do_human_parts:
            # Find images which have human parts
            self.has_human_parts = []
            for ii in range(len(self.images)):
                if self.human_parts_category in self.part_obj_dict[self.im_ids[ii]]:
                    self.has_human_parts.append(1)
                else:
                    self.has_human_parts.append(0)
            # If the other tasks are disabled, select only the images that contain human parts
            if not self.do_edge and not self.do_semseg and not self.do_sal and not self.do_normals:
                print('Ignoring images that do not contain human parts')
                for i_ in range(len(self.parts) - 1, -1, -1):
                    if self.has_human_parts[i_] == 0:
                        del self.im_ids[i_]
                        del self.images[i_]
                        del self.parts[i_]
                        del self.has_human_parts[i_]
                assert (len(self.images) == len(self.parts))
                print('Number of images with human parts: {:d}'.format(np.sum(self.has_human_parts)))
        self.count = 0

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

        if self.do_human_parts:
            _human_parts, _ = self._load_human_parts(index)
            assert np.shape(_img)[0:2] == np.shape(_human_parts)[0:2]
            sample['labels']['human_parts'] = _human_parts

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            assert np.shape(_img)[0:2] == np.shape(_semseg)[0:2]
            sample['labels']['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals_distilled(index)
            assert np.shape(_img)[0:2] == np.shape(_normals)[0:2]
            sample['labels']['normals'] = _normals

        if self.do_sal:
            _sal = self._load_sal_distilled(index)
            assert np.shape(_img)[0:2] == np.shape(_sal)[0:2]
            sample['labels']['sal'] = _sal

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
            if key in {'semseg', 'human_parts'}:
                sample['labels'][key] = torch.from_numpy(np.array(target)).long()
            elif key in {'sal', 'edge'}:
                sample['labels'][key] = torch.from_numpy(np.array(target)).float()
            elif key in {'normals'}:
                target = np.swapaxes(np.swapaxes(np.array(target), 1, 2), 0, 1)
                sample['labels'][key] = torch.from_numpy(target).float()
        return sample

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

    def _check_preprocess_parts(self):
        _obj_list_file = self.parts_file
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.part_obj_dict = json.load(open(_obj_list_file, 'r'))
            return list(np.sort([str(x) for x in self.part_obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.parts)):
            # Read object masks and get number of objects
            if ii % 100 == 0:
                print("Processing image: {}".format(ii))
            part_mat = sio.loadmat(self.parts[ii])
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for jj in range(n_obj):
                obj_cat = int(part_mat['anno'][0][0][1][0][jj][1])
                _cat_ids.append(obj_cat)
                obj_counter += 1

            self.part_obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.part_obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(
                    ',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.part_obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing for parts finished')

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_edge(self, index):
        # Read Target object
        _tmp = sio.loadmat(self.context[index])
        _edge = cv2.Laplacian(_tmp['LabelMap'], cv2.CV_64F)
        _edge = thin(np.abs(_edge) > 0).astype(np.float32)
        return _edge

    def _load_human_parts(self, index):
        if self.has_human_parts[index]:

            # Read Target object
            _part_mat = sio.loadmat(self.parts[index])['anno'][0][0][1][0]

            _inst_mask = _target = None

            for _obj_ii in range(len(_part_mat)):

                has_human = _part_mat[_obj_ii][1][0][0] == self.human_parts_category
                has_parts = len(_part_mat[_obj_ii][3]) != 0

                if has_human and has_parts:
                    if _inst_mask is None:
                        _inst_mask = _part_mat[_obj_ii][2].astype(np.float32)
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        _inst_mask = np.maximum(_inst_mask, _part_mat[_obj_ii][2].astype(np.float32))

                    n_parts = len(_part_mat[_obj_ii][3][0])
                    for part_i in range(n_parts):
                        cat_part = str(_part_mat[_obj_ii][3][0][part_i][0][0])
                        mask_id = self.cat_part[str(self.human_parts_category)][cat_part]
                        mask = _part_mat[_obj_ii][3][0][part_i][1].astype(bool)
                        _target[mask] = mask_id

            if _target is not None:
                _target, _inst_mask = _target.astype(np.float32), _inst_mask.astype(np.float32)
            else:
                shape = np.shape(sio.loadmat(self.parts[index])['anno'][0][0][1][0][0][2])
                _target, _inst_mask = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
            return _target, _inst_mask

        else:
            shape = np.shape(sio.loadmat(self.parts[index])['anno'][0][0][1][0][0][2])
            # When the output has no humans, we use outoput as 255 (equal to loss function ignore index)
            return np.ones(shape, dtype=np.float32)*255, np.ones(shape, dtype=np.float32)*255

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        return _semseg

    def _load_normals_distilled(self, index):
        _tmp = np.array(Image.open(self.normals[index])).astype(np.float32)
        _tmp = 2.0 * _tmp / 255.0 - 1.0

        labels = sio.loadmat(self.context[index])
        labels = labels['LabelMap']

        _normals = np.zeros(_tmp.shape, dtype=np.float)
        for x in np.unique(labels):
            if x in self.normals_valid_classes:
                _normals[labels == x, :] = _tmp[labels == x, :]
        return _normals

    def _load_sal_distilled(self, index):
        _sal = np.array(Image.open(self.sals[index])).astype(np.float32) / 255.
        _sal = (_sal > 0.5).astype(np.float32)
        return _sal
