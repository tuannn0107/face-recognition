from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time
import sys
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from utils import constants
import os
from facenet import face_net
from facenet.align import detect_face
from datetime import datetime

input_video = "E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Video\\Camera\\HuynhVanTien.mp4"
output_dir = "E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Video\\Camera\\RawImages1\\Recognition\\Frame_Video\\TienHuynh"
output_dir_face = "E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Video\\Camera\\RawImages1\\Recognition\\Face_video"
modeldir = constants.CLASSIFIER_MODEL
classifier_filename = constants.CLASSIFIER_FILE_VIDEO
npy = ''
train_img = constants.CLASSIFIER_DATA_DIR

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = constants.FACE_REG_MINSIZE  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = constants.FACE_REG_MARGIN
        frame_interval = 3
        batch_size = 1000
        image_size = 160
        input_image_size = 160

        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading Modal')
        face_net.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        video_capture = cv2.VideoCapture(input_video)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output/' + input_video.split('/')[-1], fourcc, 25.0, (width, height))
        c = 0
        if os.path.exists(output_dir):
            os.remove(output_dir)
        os.makedirs(output_dir)

        if os.path.exists(output_dir_face):
            os.remove(output_dir_face)
        os.makedirs(output_dir_face)

        print('Start Recognition')
        prevTime = 0
        number_face_recognited = 0
        number_frame = 0
        while True:
            ret, frame = video_capture.read()
            if ret == False:
                break
            # frame = cv2.resize(frame, (650, 650), fx=0.5, fy=0.5)  # resize frame (optional)

            number_frame += 1

            # curTime = time.time() + 1  # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = face_net.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                # if nrof_faces > 0:
                    # print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    number_face_recognited += nrof_faces
                    det = bounding_boxes[:, 0:4]
                    det_arr = []
                    img_size = np.asarray(frame.shape)[0:2]
                    # Recognition
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    result_names = 'unknown'
                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue

                        cropped_data = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        write_data = misc.imresize(cropped_data, (constants.ALIGN_IMAGE_SIZE, constants.ALIGN_IMAGE_SIZE), interp='bilinear')

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
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # print("predictions")

                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face
                        # print(best_class_probabilities)
                        if best_class_probabilities > 0:

                            # plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 15
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    cv2.putText(frame, '{0}_{1}'.format(round(best_class_probabilities[0],2), result_names), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=1)

                        output_filename_face = os.path.expanduser(output_dir_face + '/{0}_{1}_{2}_{3}_{4}.jpg'.format(round(best_class_probabilities[0],2), result_names, number_frame, number_face_recognited, datetime.strftime(datetime.now(), '%Y%m%d%H%M%S%f')))
                        cv2.imwrite(output_filename_face, write_data)
                output_filename = os.path.expanduser(output_dir + '/{0}_{1}_{2}.jpg'.format(number_frame, nrof_faces, datetime.strftime(datetime.now(), '%Y%m%d%H%M%S%f')))
                cv2.imwrite(output_filename, frame)

            # c+=1
            out.write(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('{0} face recognited.'.format(number_face_recognited))
        print('{0} frame captured.'.format(number_frame))

        video_capture.release()
        out.release()
cv2.destroyAllWindows()
