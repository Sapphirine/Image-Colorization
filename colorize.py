from keras.preprocessing import image as image_utils
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
import os
import cv2

print("*** Loading image")
folder = 'D:/Luis/Documents/mirflickr25k/train'
listing = os.listdir(folder)
directory = folder + '/train_x_1.mat'
image = sio.loadmat(directory)
im_bw = image['im_in']
print(im_bw.shape)
im_in = np.zeros((224,224,3))
im_in[:,:,0] = im_bw
im_in[:,:,1] = im_bw
im_in[:,:,2] = im_bw
im_in = np.expand_dims(im_in, axis=0)

print("*** Loading model and weights")
json_file = open('model_color.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_weights.h5")
 
print("*** Colorizing")
im_out = model.predict(im_in, batch_size=1, verbose=1)
sio.savemat('color_pic.mat', dict(im_out=im_out))
sio.savemat('original_pic.mat', dict(image=image))