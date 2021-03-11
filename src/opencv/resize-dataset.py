from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
from facenet import face_net

def main(data_dir, output_dir, resize_scale):
    output_dir = output_dir + '\\' + '{0}'.format(resize_scale)
    dataset = face_net.get_dataset(data_dir)
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        for image_path in cls.image_paths:
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            fileex = os.path.splitext(os.path.split(image_path)[1])[1]
            output_filename = os.path.join(output_class_dir, filename + '.' + fileex)

            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            scale_percent = resize_scale  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_filename, resized)

    print('The output is ' + output_dir)
    print("Finish!!!!")


if __name__ == '__main__':
    main("C:\\Users\\TuanNN\\Downloads\\lfw\\lfw", "E:\\Document\\DeepLearning\\DataSet\\Resize\\Result", 50)


