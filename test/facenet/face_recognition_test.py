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

def main(test_dir, data_dir, model_dir, classifier_file):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=constants.GPU_MEMORY_FRACTION_DEFAULT)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            minsize = constants.FACE_REG_MINSIZE  # minimum size of face
            threshold = constants.ALIGN_THRESHOLD  # three steps's threshold
            factor = constants.ALIGN_FACTOR  # scale factor
            image_size = 160
            input_image_size = 160

            human_names = os.listdir(data_dir)
            human_names.sort()

            print('Loading feature extraction model')
            face_net.load_model(model_dir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(classifier_file)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            c = 0
            print('Start Recognition!')
            dataset = face_net.get_dataset(test_dir)
            number_of_face_recognition = 0
            for cls in dataset:
                for image_path in cls.image_paths:
                    frame = cv2.imread(image_path, 0)
                    if frame.ndim == 2:
                        frame = face_net.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('Face is too close {0}'.format(image_path))
                                break

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = face_net.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = face_net.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            # print(best_class_indices)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                            # plot result idx under box
                            for H_i in human_names:
                                # print(H_i)
                                if human_names[best_class_indices[0]] == H_i \
                                        and H_i in image_path:
                                    print('{0} : {1}'.format(best_class_probabilities, image_path))
                                    number_of_face_recognition = number_of_face_recognition + 1
                    else:
                        print('Unable to recognition {0}'.format(image_path))

    print("Finish!!!!")
    print('Number face detected {0}'.format(number_of_face_recognition))


if __name__ == '__main__':
    main('E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Recognition\\HeadPoseImageDatabase',
         'E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Recognition\\HeadPoseImageDatabase', constants.CLASSIFIER_MODEL, constants.CLASSIFIER_FILE)


