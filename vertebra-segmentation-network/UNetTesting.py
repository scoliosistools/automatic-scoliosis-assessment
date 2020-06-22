import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os
import cv2
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib as mpl

IMG_SIZE_X = 128
IMG_SIZE_Y = 256

def tversky_loss(y_true, y_pred, beta=0.5):
    def loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

    return loss(y_true, y_pred)

def roi2edge(roi):
    edge = np.zeros((IMG_SIZE_Y, IMG_SIZE_X))
    for k in range(IMG_SIZE_Y):
        for j in range(IMG_SIZE_X):
            if roi[k,j] == 1:
                if not (roi[k+1,j]==1 and roi[k-1,j]==1 and roi[k,j+1]==1 and roi[k,j-1]==1 and roi[k+1,j+1]==1 and roi[k+1,j-1]==1 and roi[k-1,j+1]==1 and roi[k-1,j-1]==1):
                        edge[k,j] = 1
    return edge

def create_test_im_datasets(im_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y):
    fn_data = genfromtxt(fn_dir, delimiter=',', dtype=str)
    im_data = []

    for filename in fn_data:
        for img in os.listdir(im_dir):
            if img == filename:
                img_array = cv2.imread(os.path.join(im_dir, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE_X, IMG_SIZE_Y))
                im_data.append([new_array])

    im_data = np.array(im_data).reshape(-1, IMG_SIZE_Y, IMG_SIZE_X, 1)
    im_data = im_data / 255.0
    return im_data


test_im_dir = "../data/boostnet_labeldata/data/test"
test_fn_dir = "../data/boostnet_labeldata/labels/test/filenames.csv"

X = create_test_im_datasets(test_im_dir, test_fn_dir, IMG_SIZE_X, IMG_SIZE_Y)


model_vertebrae = tf.keras.models.load_model("G:/My Drive/GitHub/automatic-scoliosis-assessment/models/VertebraSegmentationNetwork-1585184318_beta0.7val0.15", compile=False)
model_vertebrae.compile(optimizer='adam', loss=tversky_loss, metrics=['accuracy'])
Y_predicted = model_vertebrae.predict(X)

kernel = np.ones((2,2),np.uint8)

count = 0
plot_count = 1
fig = plt.figure()
plt.title("Predictions")
for j in range(128):
    # taking a sample
    if count % 14 == 0:
        plt.subplot(2, 5, plot_count)
        plt.imshow(X[j, :, :, 0], cmap="gray")
        # roi = Y_predicted_spacing[j, :, :, 0]
        # edge = roi2edge(roi)
        gradient = cv2.morphologyEx(Y_predicted[j, :, :, 0], cv2.MORPH_GRADIENT, kernel)
        plt.imshow(gradient, 'inferno', interpolation='none', alpha=0.3)
        plot_count += 1
    count += 1
plt.show()

# ***** to save spineMasks
fn_dir = "../data/boostnet_labeldata/labels/test/filenames.csv"
fn_data = genfromtxt(fn_dir, delimiter=',', dtype=str)
for k in range(128):
    name = '../data/PredictionsVsGroundTruth/SpineMasks/'+fn_data[k]
    plt.imsave(name, Y_predicted[k, :, :, 0], cmap="gray")


