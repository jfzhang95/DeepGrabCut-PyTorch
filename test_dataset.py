import socket
from datetime import datetime
import os
import glob
from collections import OrderedDict
import scipy.misc as sm
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

from dataset import custom_transforms as tr




# Custom includes
from dataset import pascal, pascalvoc


image_width = 400
image_height = 400



voc_train = pascal.PascalVocDataset(image_size=(image_width, image_height),
                                    split='train',
                                    transform=pascalvoc.ToTensor())

composed_transforms_tr = transforms.Compose([
    tr.RandomHorizontalFlip(),
    tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
    tr.CropFromMask(crop_elems=('image', 'gt'), relax=relax_crop, zero_pad=zero_pad_crop),
    tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
    tr.ExtremePoints(sigma=10, pert=5, elem='crop_gt'),
    tr.ToImage(norm_elem='extreme_points'),
    tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
    tr.ToTensor()])





