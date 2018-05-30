import cv2
import numpy as np
import os
import torch


from torch.nn.functional import upsample
from dataset import utils
import networks.deeplab_resnet as resnet

from glob import glob
from copy import deepcopy


drawing = False
start = False

# TODO: import test image path
image_list = glob(os.path.join('ims/', '*.'+'jpg'))
image = cv2.imread(image_list[0])
img_shape = (450, 450)
image = utils.fixed_resize(image, img_shape).astype(np.uint8)

w = img_shape[0]
h = img_shape[1]
output = np.zeros((w, h, 3), np.uint8)
thres = 0.8

left = 0xFFF
right = 0
up = 0xFFF
down = 0

gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
print("Initializing weights from: {}".format(os.path.join('run', 'run_0', 'models', 'deepgc_pascal_epoch-99.pth')))
state_dict_checkpoint = torch.load(os.path.join('run', 'run_0', 'models', 'deepgc_pascal_epoch-99.pth'),
                                   map_location=lambda storage, loc: storage)

net.load_state_dict(state_dict_checkpoint)
net.eval()
net.to(device)


def interactive_drawing(event, x, y, flag, params):
    global xs, ys, ix, iy, drawing, image, output, left, right, up, down

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        xs, ys = x, y
        left = min(left, x)
        right = max(right, x)
        up = min(up, y)
        down = max(down, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            cv2.line(image, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.line(output, (ix, iy), (x, y), (255, 255, 255), 1)
            ix = x
            iy = y
            left = min(left, x)
            right = max(right, x)
            up = min(up, y)
            down = max(down, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.line(output, (ix, iy), (x, y), (255, 255, 255), 1)
        ix = x
        iy = y
        cv2.line(image, (ix, iy), (xs, ys), (0, 0, 255), 2)
        cv2.line(output, (ix, iy), (xs, ys), (255, 255, 255), 1)
    return x, y

def main():
    global image, output
    cv2.namedWindow('draw', flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('draw', interactive_drawing)

    image_idx = 0
    while(1):
        cv2.imshow('draw', image)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            # print(k)
            pass
        if k == 100: # D
            cv2.imwrite('./' + str(image_idx) + 'out.png', image)
        if k == 115:
            global left, right, up, down
            left = 0xFFF
            right = 0
            up = 0xFFF
            down = 0
            drawing = False  # true if mouse is pressed
            image = cv2.imread(image_list[image_idx])
            image = utils.fixed_resize(image, (450, 450)).astype(np.uint8)
            sp = image.shape
            w = sp[0]
            h = sp[1]
            output = np.zeros((w, h, 3), np.uint8)
            while (1):
                cv2.imshow('draw', image)
                k = cv2.waitKey(1) & 0xFF
                if k == 32:
                    break
                if k == 27:
                    image = cv2.imread(image_list[image_idx])
                    image = utils.fixed_resize(image, (450, 450)).astype(np.uint8)
                    output = np.zeros((w, h, 3), np.uint8)

            tmp = (output[:, :, 0] > 0).astype(np.uint8)
            tmp_ = deepcopy(tmp)
            fill_mask = np.ones((tmp.shape[0] + 2, tmp.shape[1] + 2))
            fill_mask[1:-1, 1:-1] = tmp_
            fill_mask = fill_mask.astype(np.uint8)
            cv2.floodFill(tmp_, fill_mask, (int((left + right) / 2), int((up + down) / 2)), 5)
            tmp_ = tmp_.astype(np.int8)

            output = cv2.resize(output, img_shape)

            tmp_ = tmp_.astype(np.int8)
            tmp_[tmp_ == 5] = -1  # pixel inside bounding box
            tmp_[tmp_ == 0] = 1  # pixel on and outside bounding box

            tmp = (tmp == 0).astype(np.uint8)

            dismap = cv2.distanceTransform(tmp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)  # compute distance inside and outside bounding box
            dismap = tmp_ * dismap + 128

            dismap[dismap > 255] = 255
            dismap[dismap < 0] = 0
            dismap = dismap

            dismap = utils.fixed_resize(dismap, (450, 450)).astype(np.uint8)

            dismap = np.expand_dims(dismap, axis=-1)

            image = image[:, :, ::-1] # change to rgb
            merge_input = np.concatenate((image, dismap), axis=2).astype(np.float32)
            inputs = torch.from_numpy(merge_input.transpose((2, 0, 1))[np.newaxis, ...])

            # Run a forward pass
            inputs = inputs.to(device)
            outputs = net.forward(inputs)
            outputs = upsample(outputs, size=(450, 450), mode='bilinear', align_corners=True)
            outputs = outputs.to(torch.device('cpu'))

            prediction = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
            prediction = 1 / (1 + np.exp(-prediction))
            prediction = np.squeeze(prediction)
            prediction[prediction>thres] = 255
            prediction[prediction<=thres] = 0

            prediction = np.expand_dims(prediction, axis=-1).astype(np.uint8)
            image = image[:, :, ::-1] # change to bgr
            display_mask = np.concatenate([np.zeros_like(prediction), np.zeros_like(prediction), prediction], axis=-1)
            image = cv2.addWeighted(image, 0.9, display_mask, 0.5, 0.1)

        if k == 99:
            break

        if k == 110:
            image_idx += 1
            if image_idx >= len(image_list):
                print('Already the last image. Starting from the beginning.')
                image_idx = 0
            image = cv2.imread(image_list[image_idx])
            image = utils.fixed_resize(image, (450, 450)).astype(np.uint8)
            sp = image.shape
            w = sp[0]
            h = sp[1]
            output = np.zeros((w, h, 3), np.uint8)

        if k == 112:
            image_idx -= 1
            if image_idx < 0:
                print('Reached the first image. Starting from the end.')
                image_idx = len(image_list)-1
            image = cv2.imread(image_list[image_idx])
            image = utils.fixed_resize(image, (450, 450)).astype(np.uint8)
            sp = image.shape
            w = sp[0]
            h = sp[1]
            output = np.zeros((w, h, 3), np.uint8)


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
