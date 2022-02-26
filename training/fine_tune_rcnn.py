import time
start_time = time.time()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from imutils import paths
# import matplotlib.pyplot as plt
import numpy as np
import logging
from constants import *
# import argparse
# import pickle
# import os

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	#help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())


# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class labels
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Loading Images started')

imagePaths = list(paths.list_images(RESOURCE_BASE_PATH))
data = []
labels = []

for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	
	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=INPUT_DIMS)
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info( 'convert the data and labels to NumPy arrays')
labels = np.array(labels)
data = np.array(data, dtype="float32")
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info( 'converted')

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20,
                                                  stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
# baseModel.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Compiling Model')
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Training head of the network')
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Evaluating Network')
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
# predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
# print(classification_report(testY.argmax(axis=1), predIdxs,
# 	target_names=lb.classes_))

# serialize the model to disk
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Saving Mask Detector Model')

model.save(CUSTOM_MODEL_PATH, save_format="h5")

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info("Execution Time Of fine_tune_rcnn.py: %s seconds " % (time.time() - start_time))
