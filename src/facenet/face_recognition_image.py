from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from facenet import face_net
from facenet.align import detect_face
from utils import constants


def main(image_path, data_dir, model_dir, classifier_file):
    npy = ''
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=constants.GPU_MEMORY_FRACTION_DEFAULT)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            minsize = constants.FACE_REG_MINSIZE  # minimum size of face
            threshold = constants.ALIGN_THRESHOLD  # three steps's threshold
            factor = constants.ALIGN_FACTOR  # scale factor
            frame_interval = 3
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
            frame = cv2.imread(image_path, 0)
            time_f = frame_interval

            if c % time_f == 0:
                if frame.ndim == 2:
                    frame = face_net.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Face Detected: %d' % nrof_faces)

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
                            print('face is too close')
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
                        print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        # print(best_class_indices)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print(best_class_probabilities)
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face

                        # plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] - 10
                        print(i, 'Result Indices: ', best_class_indices[0], ' : ', 'Face detected of : {0}'.format(human_names[best_class_indices[0]]))
                        print(human_names)
                        for H_i in human_names:
                            # print(H_i)
                            if human_names[best_class_indices[0]] == H_i and best_class_probabilities >= constants.FACE_REG_POSSIBILITY:
                                result_names = human_names[best_class_indices[0]]
                                cv2.putText(frame, str(i) + ': ' + result_names, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN,
                                            1, (0, 0, 255), thickness=1, lineType=1)
                        print('------------------')
                else:
                    print('Unable to align')
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)
            cv2.imshow('Image', frame)
            cv2.imwrite('output/' + image_path.split('/')[-1], frame)
            if cv2.waitKey(1000000) & 0xFF == ord('q'):
                sys.exit("Thanks")


if __name__ == '__main__':
    # image_path_test = "E:\\Document\\DeepLearning\\DataSet\\Employees_Test\\TuanNN\\aTest.jpg"
    image_path_test = "E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Video\\Friends\\RawImages4\\Friends\\0_61_20190927195030584080.jpg"
    main(image_path_test, constants.CLASSIFIER_DATA_DIR, constants.CLASSIFIER_MODEL, constants.CLASSIFIER_FILE)
    cv2.destroyAllWindows()
