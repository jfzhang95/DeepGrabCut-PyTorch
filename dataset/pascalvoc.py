import os
import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, sampler
from natsort import natsorted
from mypath import Path

class PascalVocDataset(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self, base_dir=Path.db_root_dir('pascal'), image_size=None, phase='train', transform=None):
        """

        :param base_dir: path to DAVIS dataset directory
        :param image_size: (width, height) tuple to resize the image
        :param year: which train/val split of DAVIS to use
        :param phase: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_size = image_size
        self._images_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._annotations_dir = os.path.join(self._base_dir, 'SegmentationGT')
        self._dismaps_dir = os.path.join(self._base_dir, 'DistanceMaps')
        self._transform = transform

        self.sequences = []
        with open(os.path.join(self._base_dir, '{}.txt'.format(phase)), 'r') as f:
            self.sequences += [seq.strip() for seq in f.readlines()]

        # Store all the image paths, annotation paths, frame numbers and sequence labels
        self._object_data = []
        for seq in self.sequences:
            self._object_data.append([seq.split()[0], seq.split()[1]])



    def __len__(self):
        return len(self._object_data)

    def __getitem__(self, idx):
        # Load the <idx>th image and annotation and return a sample
        image_name, annotation_name = self._object_data[idx]
        map_choice = np.random.randint(1, 17)
        dismap_name = annotation_name + '_{}'.format(str(map_choice))

        image_path = os.path.join(self._images_dir, image_name+'.jpg')
        annotation_path = os.path.join(self._annotations_dir, annotation_name+'.png')
        dismap_path = os.path.join(self._dismaps_dir, dismap_name+'.png')

        # image = cv2.imread(image_path)
        # annotation = cv2.imread(annotation_path)[:, :, 0]
        # dismap = cv2.imread(dismap_path)[:, :, 0]

        image = Image.open(image_path).convert('RGB')
        annotation = Image.open(annotation_path).convert('L')
        dismap = Image.open(dismap_path).convert('L')


        if self._image_size is not None:
            # image = cv2.resize(image, self._image_size)
            # annotation = cv2.resize(annotation, self._image_size)
            # dismap = cv2.resize(dismap, self._image_size)
            image = image.resize(self._image_size)
            annotation = annotation.resize(self._image_size)
            dismap = dismap.resize(self._image_size)

        image = np.asarray(image)
        annotation = np.asarray(annotation)
        dismap = np.asarray(dismap)

        input = np.concatenate((image, np.expand_dims(dismap, -1)), axis=-1)



        # # normalize the image
        # image = np.asarray(image)
        # image = (image - image.mean()) / image.std()
        #
        # # convert annotation to binary image
        # annotation = np.asarray(annotation).copy()
        # annotation[annotation > 0] = 1

        sample = {
            'input': input,
            'annotation': annotation
        }
        if self._transform:
            sample = self._transform(sample)
        return sample



# Transforms
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'input': torch.from_numpy(sample['input'].transpose((2, 0, 1))),
                'annotation': torch.from_numpy(sample['annotation'].astype(np.uint8))
        }
