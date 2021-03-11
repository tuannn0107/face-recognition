"""===================================
    Argument default value for Facenet
==================================="""
# Parameters default value for facenet
LOGS_BASE_DIR_DEFAULT = 'logs/facenet'
MODELS_BASE_DIR_DEFAULT = 'models/facenet'
GPU_MEMORY_FRACTION_DEFAULT = 1.0
PRETRAINED_MODEL_DEFAULT = ''  # ''E:/Document/DeepLearning/PretrainModel/facenet/20180402-114759'
DATA_DIR_DEFAULT = 'E:\\Document\\DeepLearning\\DataSet\\Friends\\Database'
MODEL_DEF_DEFAULT = 'models.inception_resnet_v1'
MAX_NROF_EPOCHS_DEFAULT = 50
BATCH_SIZE = 10
IMAGE_SIZE = 160
PEOPLE_PER_BATCH_DEFAULT = 3
IMAGES_PER_PERSON_DEFAULT = 2
EPOCH_SIZE_DEFAULT = 50
ALPHA_DEFAULT = 0.2
EMBEDDING_SIZE_DEFAULT = 128
RANDOM_CROP_DEFAULT = ''
RANDOM_FLIP_DEFAULT = ''
KEEP_PROBABILITY_DEFAULT = 1.0
WEIGHT_DECAY_DEFAULT = 0.0
OPTIMIZER_DEFAULT = 'ADAGRAD'
LEARNING_RATE_DEFAULT = 0.1
LEARNING_RATE_DECAY_EPOCHS_DEFAULT = 100
LEARNING_RATE_DECAY_EPOCHS_FACTOR = 100
MOVING_AVERAGE_DECAY_DEFAULT = 0.9999
SEED_DEFAULT = 666
LEARNING_RATE_SCHEDULE_FILE_DEFAULT = 'data/learning_rate_schedule.txt'

# Parameters for validation on LFW
LFW_PAIRS_DEFAULT = 'data/pairs.txt'
LFW_DIR_DEFAULT = ''
LFW_NROF_FOLDS_DEFAULT = 10
LFW_NROF_PREPROCESS_THREAD_DEFAULT = 4


"""==================================
    Argument default value for Image compare
=================================="""
COMPARE_MODEL_DEFAULT = ''
COMPARE_IMAGE_FILES_DEFAULT = ''
COMPARE_IMAGE_SIZE_DEFAULT = 160
COMPARE_MARGIN_DEFAULT = 44
COMPARE_MODEL_DIR_FACE_DETECTION_DEFAULT = 'E:/Document/DeepLearning/PretrainModel/facenet/20180402-114759'
COMPARE_IMAGE_PATH_FOR_COMPARE = 'E:\\Document\\DeepLearning\\DataSet\\Employees'


"""==================================
    Argument default value for face align
=================================="""
#ALIGN_INPUT_DIR = 'E:\\Document\\DeepLearning\\DataSet\\Images\\Test_Align\\Image'
# ALIGN_INPUT_DIR = 'E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Recognition\\HeadPoseImageDatabase'
ALIGN_INPUT_DIR = 'E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Video\\Camera\\RawImages1\\Detection\\Frame_Video'
#ALIGN_OUTPUT_DIR = 'E:\\Document\\DeepLearning\\DataSet\\Images\\Test_Align\\Aligned'
# ALIGN_OUTPUT_DIR = 'E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Recognition\\HeadPoseImageDatabase_aligned'
ALIGN_OUTPUT_DIR = 'E:\\Document\\DeepLearning\\DataSet\\ForTesting\\Video\\Camera\\RawImages1\\Detection\\Frame_Video_aligned'
ALIGN_IMAGE_SIZE = 160
ALIGN_DETECT_MULTIPLE_FACES = True
ALIGN_RANDOM_ORDER = ''
ALIGN_THRESHOLD = [0.6, 0.7, 0.7]
ALIGN_FACTOR = 0.709


"""==================================
    Argument default value for classifier
=================================="""
CLASSIFIER_MODE = 'TRAIN'
CLASSIFIER_MODE_CLASSIFY = 'CLASSIFY'
#CLASSIFIER_DATA_DIR = 'E:\\Document\\DeepLearning\\DataSet\\Employees'
CLASSIFIER_DATA_DIR = 'E:\\Document\\DeepLearning\\DataSet\\Friends\\Aligned'
#CLASSIFIER_DATA_DIR = 'E:\\Document\\DeepLearning\\DataSet\\Images\\Test_Align\\Aligned'
CLASSIFIER_OUT_DIR = 'E:/Document/DeepLearning/PretrainModel/facenet/20180402-114759'
CLASSIFIER_MODEL = 'E:/Document/DeepLearning/PretrainModel/facenet/20180402-114759'
#CLASSIFIER_FILE = 'E:/Document/DeepLearning/PretrainModel/facenet/20180402-114759/classifier_20190910220644.pkl'
CLASSIFIER_FILE = 'E:/Document/DeepLearning/PretrainModel/facenet/20180402-114759/classifier_20190927083812.pkl'
CLASSIFIER_FILE_VIDEO = 'E:/Document/DeepLearning/PretrainModel/facenet/20180402-114759/classifier_20190928092737_Friends.pkl'
CLASSIFIER_USE_SPLIT_DATASET = ''
CLASSIFIER_TEST_DATA_DIR = ''
CLASSIFIER_BATCH_SIZE = 100
CLASSIFIER_SEED = 666
CLASSIFIER_MIN_NROF_IMAGES_PER_CLASS = 2
CLASSIFIER_NROF_TRAIN_IMAGES_PER_CLASS = 4

"""==================================
    Argument default value for face recognition_image
=================================="""
FACE_REG_POSSIBILITY = 0.4
FACE_REG_MINSIZE = 30
FACE_REG_MARGIN = 40


"""==================================
    Other constants
=================================="""
DATASET_EMPLOYEE_PATH = "E:/Document/DeepLearning/DataSet/Employees"
MIN_DIST_INIT = 100
MIN_DIST = 0.4

"""==================================
    GUI
=================================="""
GUI_WIDTH = 650
GUI_HEIGHT = 400

