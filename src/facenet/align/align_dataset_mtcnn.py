"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import os
import argparse
import tensorflow as tf
import numpy as np
from facenet import face_net
from facenet.align import detect_face
import random
from time import sleep
from utils import constants


def main(data_dir, output_dir):
    sleep(random.random())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, ' '.join('argument default'))
    dataset = face_net.get_dataset(data_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=constants.GPU_MEMORY_FRACTION_DEFAULT)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = constants.FACE_REG_MINSIZE  # minimum size of face
    threshold = constants.ALIGN_THRESHOLD  # three steps's threshold
    factor = constants.ALIGN_FACTOR  # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)

    face_detected_list = []

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if constants.ALIGN_RANDOM_ORDER:
        random.shuffle(dataset)
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if constants.ALIGN_RANDOM_ORDER:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    error_message = '{}: {}'.format(image_path, e)
                    print(error_message)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        face_detected_list.append('Unable to align {0}'.format(image_path))
                        # text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = face_net.to_rgb(img)
                    img = img[:, :, 0:3]

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            if constants.ALIGN_DETECT_MULTIPLE_FACES:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det_arr.append(det[index, :])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - constants.COMPARE_MARGIN_DEFAULT / 2, 0)
                            bb[1] = np.maximum(det[1] - constants.COMPARE_MARGIN_DEFAULT / 2, 0)
                            bb[2] = np.minimum(det[2] + constants.COMPARE_MARGIN_DEFAULT / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + constants.COMPARE_MARGIN_DEFAULT / 2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            scaled = misc.imresize(cropped, (constants.ALIGN_IMAGE_SIZE, constants.ALIGN_IMAGE_SIZE), interp='bilinear')
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if constants.ALIGN_DETECT_MULTIPLE_FACES:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            misc.imsave(output_filename_n, scaled)
                            face_detected_list.append('%s --- BOX [%d , %d , %d , %d]\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        face_detected_list.append('Unable to align {0}'.format(image_path))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    return nrof_images_total, nrof_successfully_aligned, face_detected_list


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str,
                        help='Directory with unaligned images.', default=constants.ALIGN_INPUT_DIR)
    parser.add_argument('output_dir', type=str,
                        help='Directory with aligned face thumbnails.', default=constants.ALIGN_OUTPUT_DIR)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=constants.ALIGN_IMAGE_SIZE)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=constants.COMPARE_MARGIN_DEFAULT)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=constants.GPU_MEMORY_FRACTION_DEFAULT)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=constants.ALIGN_DETECT_MULTIPLE_FACES)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(constants.ALIGN_INPUT_DIR, constants.ALIGN_OUTPUT_DIR)
