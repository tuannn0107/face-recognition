import numpy as np
import pickle

import os
import random

import cv2

DATADIR = "E:\\Document\\DeepLearning\\DataSet\\PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 30


def create_training_data(data_dir, categories, img_size):
    output = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                output.append([new_array, class_num])
            except Exception as e:
                pass

    return output


training_data = create_training_data(DATADIR, CATEGORIES, IMG_SIZE)

random.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("x.pkl", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pkl", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("x.pkl", "rb")
x = pickle.load(pickle_in)







