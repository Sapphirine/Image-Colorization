from vgg16_mod import VGG16
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from imutils import paths
from imagenet_utils import preprocess_input, decode_predictions
from math import floor
import numpy as np
import scipy.io as sio
import cv2
import os

model = VGG16(include_top=True, weights=None, input_tensor=None)
folder = 'D:/Luis/Documents/mirflickr25k/train'
listing = os.listdir(folder)

print("*** Loading pictures into data and label arrays")
data = []
labels = []
for imagePath in listing:
    directory = folder + '/'+ imagePath
    im = sio.loadmat(directory)
    if imagePath.find('y')>-1:
        im_out = im['im_out']
        label = np.ones( (im_out.shape[0], im_out.shape[1], im_out.shape[2], 2) )    
        label[:, :, :, 0] = im_out[:, :, :, 0]
        label[:, :, :, 1] = im_out[:, :, :, 1]
        labels.append(label)
    if imagePath.find('x')>-1:
        im_in = im['im_in']
        features = np.ones( (im_in.shape[0],im_in.shape[1],3) )
        features[:, :, 0] = im_in
        features[:, :, 1] = im_in
        features[:, :, 2] = im_in
        data.append(features)
    

labels = np.array(labels)
#data_scaled = np.array(data)/255
data_scaled = np.array(data)
#data_scaled = preprocess_input(data_scaled)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("*** Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data_scaled, labels, test_size=0.1, random_state=42)

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=True)
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = trainData
y_train = trainLabels
X_test = testData
y_test = testLabels

print("*** Training")
hist = model.fit(X_train, y_train, nb_epoch=10, batch_size=1, verbose = 2)
print("*** Testing")
score = model.evaluate(X_test, y_test, batch_size=1, verbose = 2)
print (score)
print("*** Saving CNN model")
model_json = model.to_json()
with open("model_color.json", "w") as json_file:
    json_file.write(model_json)
print("*** Saving weights")
model.save_weights("model_weights.h5")

