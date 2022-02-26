import os


# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
PATH_TO_PIPELINE= "C:/Users/VISHNU BHAARATH/Documents/Internship/Human Detection"

RESOURCE_BASE_PATH = os.path.sep.join([PATH_TO_PIPELINE, "training/resources_custom/classification"])
POSITIVE_PATH = os.path.sep.join([RESOURCE_BASE_PATH, "pedestrian"])
NEGATIVE_PATH = os.path.sep.join([RESOURCE_BASE_PATH, "no_pedestrian"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the path to the output model and label binarizer
CUSTOM_MODEL_PATH = os.path.sep.join([PATH_TO_PIPELINE, "training/resources_custom/custom_person_detector.h5"])

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 5
BS = 32


DIRECTORY_IMAGES = os.path.sep.join([PATH_TO_PIPELINE, 'training/resources_custom/pedestrian/images'])
IMAGE_FILES=os.listdir(DIRECTORY_IMAGES)
SORTED_IMAGE_FILES =  sorted(IMAGE_FILES)


DIRECTORY_LABELS = os.path.sep.join([PATH_TO_PIPELINE, 'training/resources_custom/pedestrian/annotations'])
LABEL_FILES=os.listdir(DIRECTORY_LABELS)
SORTED_LABEL_FILES =  sorted(LABEL_FILES)

