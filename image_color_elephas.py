from __future__ import print_function
from vgg16_mod import VGG16
from sklearn.cross_validation import train_test_split
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from math import floor
import numpy as np
import scipy.io as sio
import os
import tables
from elephas.spark_model import SparkModel, SparkMLlibModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers
from pyspark import SparkContext, SparkConf
from elephas.utils.rdd_utils import to_simple_rdd
#from elephas.ml_model import ElephasEstimator #fix keyword_only by importing pyspark, not pyspark.ml_util
#pip install --upgrade --no-deps git+git://github.com/maxpumperla/elephas to fix slice_x

# Define basic parameters
batch_size = 1
nb_classes = 10
nb_epoch = 10

# Create Spark context
conf = SparkConf().setAppName('image_color').setMaster('local[8]')
sc = SparkContext(conf=conf)


hdf5_path = "/data/project/image_color_data.hdf5"
if not os.path.isfile(hdf5_path):
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data',
                                        tables.Atom.from_dtype(np.dtype('float64')),
                                        shape=(0,224, 224, 3),
                                        filters=filters,
                                        expectedrows=25000)
    labels_storage = hdf5_file.create_earray(hdf5_file.root, 'labels',
                                            tables.Atom.from_dtype(np.dtype('float64')),
                                            shape=(0,224, 224, 50, 2),
                                            filters=filters,
                                            expectedrows=25000)

    folder = '/data/project/train'
    listing = os.listdir(folder) 

    print("*** Loading pictures into data and label arrays")
    for imagePath in listing:
        directory = folder + '/'+ imagePath
        if imagePath.find('y')>-1:
            im = sio.loadmat(directory)
            im_out = im['im_out']
            label = np.ones( (im_out.shape[0], im_out.shape[1], im_out.shape[2], 2) )
            label[:, :, :, 0] = im_out[:, :, :, 0]
            label[:, :, :, 1] = im_out[:, :, :, 1]
            labels_storage.append(label[None])
        if imagePath.find('x')>-1:
            im = sio.loadmat(directory)
            im_in = im['im_in']
            features = np.ones( (im_in.shape[0],im_in.shape[1],3) )
            features[:, :, 0] = im_in
            features[:, :, 1] = im_in
            features[:, :, 2] = im_in
            data_storage.append(features[None])
    data_scaled = data_storage
    labels = labels_storage
    hdf5_file.close()


print("*** Loading hdf5 into data and label arrays")
hdf5_file = tables.open_file(hdf5_path, mode='r')
data_scaled = hdf5_file.root.data[:]
labels = hdf5_file.root.labels[:]
hdf5_file.close()


# Define and compile a Keras model
model = VGG16(include_top=True, weights=None, input_tensor=None)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("*** Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data_scaled, labels, test_size=0.1, random_state=42)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = trainData
y_train = trainLabels
X_test = testData
y_test = testLabels

rdd = to_simple_rdd(sc,X_train,y_train)


# optional: define elephas optimizer (which tells the model how to aggregate updates on the Spark master)
#adadelta = elephas_optimizers.Adadelta()
#adagrad = elephas_optimizers.Adagrad()

spark_model = SparkModel(sc,
                         model,
                         optimizer=sgd,#adagrad,
                         frequency='epoch',
                         mode='asynchronous',
                         num_workers=2,master_optimizer=sgd)

# Train Spark model
print("*** Training Spark model...")
spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=0.1)

# Evaluate Spark model by evaluating the underlying model
score = spark_model.master_network.evaluate(testData, testLabels, verbose=2)
print('Test accuracy:', score[1])

print("*** Saving CNN model")
model_json = model.to_json()
with open("model_color.json", "w") as json_file:
    json_file.write(model_json)
print("*** Saving weights")
with open("test_model_weights.h5","w") as weights_file:
    weights_file.write(spark_model.weights)

