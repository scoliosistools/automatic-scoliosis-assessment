import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os
import cv2
import tensorflow as tf
import pickle

IMG_SIZE_X = 128
IMG_SIZE_Y = 256


def create_im_datasets(im_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y):
    # extract filenames and landmark data into arrays
    fn_data = genfromtxt(fn_dir, delimiter=',', dtype=str)
    im_data = []

    # extract images in order of filenames - same order as landmarks
    for filename in fn_data:
        idx = filename.index(".jpg")
        filenameList = []
        filenameList.append(filename[:idx] + "-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+5" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-5" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+10" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-10" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma05" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma15" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+5-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-5-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+10-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-10-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma05-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma15-mirrored" + filename[idx:])
        for name in filenameList:
            for img in os.listdir(im_dir):
                if img == name:
                    im_array = cv2.imread(os.path.join(im_dir, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(im_array, (IMG_SIZE_X, IMG_SIZE_Y))
                    im_data.append([new_array])

    # save images in array and normalise
    im_data = np.array(im_data).reshape(-1, IMG_SIZE_Y, IMG_SIZE_X, 1)
    im_data = im_data / 255.0


    return im_data


# function to generate datasets with segmentation maps of entire spinal column
def create_roi_datasets(roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y):
    # extract filenames and landmark data into arrays
    fn_data = genfromtxt(fn_dir, delimiter=',', dtype=str)
    roi_data = []

    # extract ROIs in order of filenames - same order as landmarks
    for filename in fn_data:
        idx = filename.index(".jpg")
        filenameList = []
        filenameList.append(filename[:idx] + "-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+5" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-5" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+10" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-10" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma05" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma15" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+5-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-5-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated+10-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-rotated-10-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma05-mirrored" + filename[idx:])
        filenameList.append(filename[:idx] + "-gamma15-mirrored" + filename[idx:])
        for name in filenameList:
            for roi in os.listdir(roi_dir):
                if roi == name:
                    roi_array = cv2.imread(os.path.join(roi_dir, roi), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(roi_array, (IMG_SIZE_X, IMG_SIZE_Y))
                    roi_data.append([new_array])


    # save images in array and binarize
    roi_data = np.array(roi_data).reshape(-1, IMG_SIZE_Y, IMG_SIZE_X, 1)
    roi_data[roi_data < 0.5] = 0
    roi_data[roi_data >= 0.5] = 1

    roi_data = roi_data.astype(float)
    roi_data = np.squeeze(roi_data)

    return roi_data


train_im_dir = "../data/DataAugmentation/images"
train_fn_dir = "../data/boostnet_labeldata/labels/training/filenames.csv"

# Save datasets for u-net to segment each individual vertebra
train_im_data = create_im_datasets(train_im_dir, train_fn_dir, IMG_SIZE_X, IMG_SIZE_Y)

pickle_out = open("G:/My Drive/GitHub/automatic-scoliosis-assessment/train_images.pickle", "wb")
pickle.dump(train_im_data, pickle_out)
pickle_out.close()


train_roi_dir = "../data/DataAugmentation/masks"
train_fn_dir = "../data/boostnet_labeldata/labels/training/filenames.csv"

# Save datasets for u-net to segment each individual vertebra
train_roi_data_vertebrae = create_roi_datasets(train_roi_dir, train_fn_dir, IMG_SIZE_X, IMG_SIZE_Y)

pickle_out = open("G:/My Drive/GitHub/automatic-scoliosis-assessment/train_masks.pickle", "wb")
pickle.dump(train_roi_data_vertebrae, pickle_out)
pickle_out.close()
