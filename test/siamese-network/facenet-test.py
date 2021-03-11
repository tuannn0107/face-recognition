from src.facenetpkg.training import facenet
from src.utils import constants
import os
import glob

def prepare_database_test():
    """
    Test load image from dataset
    :return:
    """
    database = facenet.prepare_database(constants.DATASET_EMPLOYEE_PATH)
    expected = len(os.listdir(constants.DATASET_EMPLOYEE_PATH))
    assert len(database) == expected


def calculate_distance_test_same_image():
    """
    Test calculate distance of 2 image
    :return:
    """
    root_path = constants.DATASET_EMPLOYEE_PATH + "_Test"
    model = facenet.FRmodel
    root_image = facenet.img_path_to_encoding(os.path.join(constants.DATASET_EMPLOYEE_PATH, "TuanNN", "01.PNG"), model)

    for employee in os.listdir(root_path):
        employee_path = os.path.join(root_path, employee, '*')
        for file in glob.glob(employee_path):
            image_to_compare_path = os.path.join(employee_path, file)
            image_to_compare = facenet.img_path_to_encoding(image_to_compare_path, model)
            dist = facenet.calculate_distance(root_image, image_to_compare)
            print('distance for image {0} is {1}'.format(file, dist))


if __name__ == "__main__":
    # prepare_database_test()
    calculate_distance_test_same_image()



