from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time
from datetime import datetime

import cv2
from facenet import face_net
import numpy as np
import tensorflow as tf
from scipy import misc

from facenet.align import detect_face
from utils import constants

modeldir = constants.CLASSIFIER_MODEL
npy = ''

def main(data_dir, min_size):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = min_size  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
        dataset = face_net.get_dataset(data_dir)

        number_sample = 0
        number_face_detected = 0
        for cls in dataset:
            for image_path in cls.image_paths:
                number_sample = number_sample + 1
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img.ndim == 2:
                    img = face_net.to_rgb(img)
                img = img[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if (nrof_faces == 0):
                    print('{0} face detected : {1}'.format(nrof_faces, image_path))
                elif(nrof_faces == 2):
                        print('{0} face detected : {1}'.format(nrof_faces, image_path))
                        number_face_detected = number_face_detected + 1
                else:
                    number_face_detected = number_face_detected + 1

    print("Finish!!!!")
    print('Number face detected {0}'.format(number_face_detected))


if __name__ == '__main__':
    main("E:\\Document\\DeepLearning\\DataSet\\ForTesting\\LightCondition", 10)


