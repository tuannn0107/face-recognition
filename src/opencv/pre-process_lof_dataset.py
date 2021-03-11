from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
from facenet import face_net


def main(data_dir, output_dir):
    dataset = face_net.get_dataset(data_dir)
    for cls in dataset:
        for i in range(len(cls.image_paths)):
            image_path = cls.image_paths[i]

            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            fileex = os.path.splitext(os.path.split(image_path)[1])[1]

            person_name = filename.split('_')[0]
            output_class_dir = os.path.join(output_dir, person_name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            output_filename = os.path.join(output_class_dir, filename + '.' + fileex)
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(output_filename, img)

    print('The output is ' + output_dir)
    print("Finish!!!!")


if __name__ == '__main__':
    main("E:\\Document\\DeepLearning\\DataSet\\ForTesting\\LightCondition", "E:\\Document\\DeepLearning\\DataSet\\ForTesting\\LightCondition_Processed")


