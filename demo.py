import cv2
import numpy as np
import os, sys
import torch

from glob import glob
from copy import deepcopy
drawing = False
start = False

# TODO: import test image path
# test_img_list =
ckpt_path = 'checkpoints/'
batch_size = 1
num_class = 2
image_list = glob(os.path.join('demo/', '*.'+'jpg'))
image = cv2.imread(image_list[0])
img_shape = (320, 320)
image = cv2.resize(image, img_shape)

w = img_shape[0]
h = img_shape[1]
output = np.zeros((w, h, 3), np.uint8)

left = 0xFFF
right = 0
up = 0xFFF
down = 0


# mouse callback function
def interactive_drawing(event, x, y, flags, param):
    global xs, ys, ix, iy, drawing, image, output, left, right, up, down

    if event == cv2.EVENT_LBUTTONDOWN:
        print('down')
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
    assert(ckpt_path != '')

    # configuration session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # define placeholder
    inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 4))
    keep_prob = tf.placeholder(tf.float32)

    # define data loader
    with tf.name_scope('DeconvNet'):
        model = DeconvNet()
        model.build(inputs, keep_prob)


    saver = tf.train.Saver()  # must be added in the end

    ''' Main '''
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    try:
        path = tf.train.get_checkpoint_state(ckpt_path)
        saver.restore(sess, path.model_checkpoint_path)
        print('-----> load from checkpoint ' + ckpt_path)
    except:
        print('unable to load checkpoint ...')
        sys.exit(0)

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
            cv2.imwrite('results/' + str(image_idx) + 'out.png', image)
        if k == 115:
            global left, right, up, down
            left = 0xFFF
            right = 0
            up = 0xFFF
            down = 0
            drawing = False  # true if mouse is pressed
            image = cv2.imread(image_list[image_idx])
            image = cv2.resize(image, img_shape)
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
                    image = cv2.resize(image, img_shape)
                    output = np.zeros((w, h, 3), np.uint8)

            output = (output[:, :, 0] > 0).astype(np.uint8)
            output[output == 1] = 10  # 10 相当于一个flag，便于调整fill_mask里的值
            output_new = deepcopy(output)
            fill_mask = np.ones((output.shape[0] + 2, output.shape[1] + 2))
            fill_mask[1:-1, 1:-1] = output_new
            fill_mask = fill_mask.astype(np.uint8)
            cv2.floodFill(output_new, fill_mask, (int((left + right) / 2), int((up + down) / 2)), 1)
            # fill_mask获得的是填充后的图 包括边缘和边缘内部， fill_mask边缘和内部的值都被填充为1
            # output边缘的值为10，其余地方的值为0
            fill_mask = fill_mask.astype(np.float32)
            fill_mask = cv2.resize(fill_mask, img_shape)

            output = cv2.resize(output, img_shape)

            fill_mask[fill_mask == 1] = -1
            fill_mask[fill_mask == 0] = 1
            fill_mask[output == 10] = 1
            output = (output == 0).astype(np.uint8) # output中边缘值为0，其余地方值为1

            dis_map = cv2.distanceTransform(output, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) # 计算边缘内部和外部到边缘的最短距离
            dis_map = fill_mask * dis_map + 128

            dis_map[dis_map > 255] = 255
            dis_map[dis_map < 0] = 0
            dis_map = dis_map.astype(np.uint8)
            # cv2.imwrite('output.png', dis_map)

            ori_image = cv2.imread(image_list[image_idx])
            # ori_shape = np.shape(ori_image)[0:2]
            resized_image = cv2.resize(ori_image, img_shape)
            dis_map = np.reshape(dis_map, np.shape(dis_map) + (1,))
            dis_map = cv2.imread('data/VOC2012/DistanceMaps/2007_000032_1_1.png')
            dismap = dis_map[:, :, 0]
            dismap = cv2.resize(dismap, img_shape)
            dismap = np.expand_dims(dismap, axis=-1)


            merge_input = np.concatenate((resized_image, dismap), axis=2)
            x_batch = np.reshape(merge_input, newshape=(1,) + np.shape(merge_input)) / 256.0
            feed_dict = {
                inputs: x_batch,
                keep_prob: 1.0
            }
            logits = model.logits
            pred = tf.cast(tf.argmax(tf.reshape(tf.nn.softmax(logits), (img_shape[0], img_shape[1], num_class)),
                             dimension=2), tf.int64)

            prediction, logits_ = sess.run([pred,logits], feed_dict=feed_dict)
            print(np.shape(logits))
            print(np.unique(prediction))
            print('prediction done!')

            red_mask = np.zeros((img_shape+(3,)))
            red_mask[:, :] = (0, 0, 255)

            display_mask = 0.4 * np.reshape(prediction, newshape=(prediction.shape+(1,)))
            # print(image.shape)
            image = (image * (1 - display_mask) + red_mask * display_mask).astype(np.uint8)

        if k == 99:
            break

        if k == 110:
            image_idx += 1
            if image_idx >= len(image_list):
                print('Already the last image. Starting from the beginning.')
                image_idx = 0
            image = cv2.imread(image_list[image_idx])
            image = cv2.resize(image, img_shape)
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
            image = cv2.resize(image, img_shape)
            sp = image.shape
            w = sp[0]
            h = sp[1]
            output = np.zeros((w, h, 3), np.uint8)


    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
