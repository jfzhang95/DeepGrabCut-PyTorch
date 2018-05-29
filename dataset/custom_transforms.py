import torch, cv2

import numpy.random as random
import numpy as np
from dataset import utils

class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=False):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        for elem in sample.keys():
            if 'meta' in elem:
                continue

            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp

        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None):
        self.resolutions = resolutions
        self.flagvals = flagvals
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())

        for elem in elems:

            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue

            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = utils.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = utils.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = utils.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = utils.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class DistanceMap(object):
    """
    Returns the distance map in a given binary mask
    v: controls the degree of rectangle variation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, v=0.15, elem='gt'):
        self.v = v
        self.elem = elem

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('DistanceMap not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            # TODO: if mask do no have any object, distance=255
            sample['distance_map'] = np.zeros(_target.shape, dtype=_target.dtype) + 255
        else:
            sample['distance_map'] = utils.distance_map(_target, self.v)

        return sample

    def __str__(self):
        return 'DistanceMap:(v='+str(self.v)+', elem='+str(self.elem)+')'


class ConcatInputs(object):

    def __init__(self, elems=('image', 'distance_map')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            res = np.concatenate((res, tmp), axis=2)

        sample['concat'] = res

        return sample

    def __str__(self):
        return 'ConcatInputs:'+str(self.elems)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem].astype(np.float32)

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp).float()

        return sample

    def __str__(self):
        return 'ToTensor'
