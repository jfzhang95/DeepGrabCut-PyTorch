import os

import torch, cv2
import random
import numpy as np
from copy import deepcopy


def distance_map(mask, pad=10, v=0.15, relax=False):
    bounds = (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)

    bbox = get_bbox(mask, pad=pad, relax=relax) # get image bounding box
    if bbox is not None:
        bbox = add_turbulence(bbox, v=v)
    else:
        dismap = np.zeros_like(mask) + 255
        return dismap

    x_min = max(bbox[0], bounds[0])
    y_min = max(bbox[1], bounds[1])
    x_max = min(bbox[2], bounds[2])
    y_max = min(bbox[3], bounds[3])

    bbox = [x_min, y_min, x_max, y_max]

    dismap = np.zeros((mask.shape[0], mask.shape[1]))
    dismap = compute_dismap(dismap, bbox)
    return dismap


def compute_dismap(dismap, bbox):
    x_min, y_min, x_max, y_max = bbox[:]

    # draw bounding box
    cv2.line(dismap, (x_min, y_min), (x_max, y_min), color=1, thickness=1)
    cv2.line(dismap, (x_min, y_min), (x_min, y_max), color=1, thickness=1)
    cv2.line(dismap, (x_max, y_max), (x_max, y_min), color=1, thickness=1)
    cv2.line(dismap, (x_max, y_max), (x_min, y_max), color=1, thickness=1)

    tmp = (dismap > 0).astype(np.uint8) # mark boundary
    tmp_ = deepcopy(tmp)

    fill_mask = np.ones((tmp.shape[0] + 2, tmp.shape[1] + 2)).astype(np.uint8)
    fill_mask[1:-1, 1:-1] = tmp_
    cv2.floodFill(tmp_, fill_mask, (int((x_min + x_max) / 2), int((y_min + y_max) / 2)), 5) # fill pixel inside bounding box

    tmp_ = tmp_.astype(np.int8)
    tmp_[tmp_ == 5] = -1 # pixel inside bounding box
    tmp_[tmp_ == 0] = 1 # pixel on and outside bounding box

    tmp = (tmp == 0).astype(np.uint8)

    dismap = cv2.distanceTransform(tmp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # compute distance inside and outside bounding box
    dismap = tmp_ * dismap + 128

    dismap[dismap > 255] = 255
    dismap[dismap < 0] = 0

    dismap = dismap.astype(np.uint8)

    return dismap


def add_turbulence(bbox, v=0.15):
    x_min, y_min, x_max, y_max = bbox[:]
    x_min_new = int(x_min + v * np.random.normal(0, 1) * (x_max - x_min))
    x_max_new = int(x_max + v * np.random.normal(0, 1) * (x_max - x_min))
    y_min_new = int(y_min + v * np.random.normal(0, 1) * (y_max - y_min))
    y_max_new = int(y_max + v * np.random.normal(0, 1) * (y_max - y_min))

    return [x_min_new, y_min_new, x_max_new, y_max_new]


def get_bbox(mask, points=None, pad=0, relax=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if relax:
        pad = 0

    x_min_bound = 0
    y_min_bound = 0
    x_max_bound = mask.shape[1] - 1
    y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return [x_min, y_min, x_max, y_max]


def fixed_resize(sample, resolution, flagval=None):
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(
            round(float(resolution) / np.min(sample.shape[:2]) * np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)

    return sample


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()
