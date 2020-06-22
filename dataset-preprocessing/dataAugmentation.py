import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os
import cv2
import tensorflow as tf
import pickle
from PIL import Image
import argparse


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def augmentimages(train_im_dir, train_fn_dir, destination):
    # extract filenames and landmark data into arrays
    fn_data = genfromtxt(train_fn_dir, delimiter=',', dtype=str)
    im_data = []

    # extract images in order of filenames - same order as landmarks
    for filename in fn_data:
        for img in os.listdir(train_im_dir):
            if img == filename:
                image = cv2.imread(os.path.join(train_im_dir, img), cv2.IMREAD_GRAYSCALE)
                original_dir = os.path.join(destination, img)
                cv2.imwrite(original_dir, image)

                idx = original_dir.index(".jpg")

                image_flipped = cv2.flip(image, 1)
                new_dir = original_dir[:idx] + "-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, 5, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated+5" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated+5-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, 10, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated+10" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated+10-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, -5, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated-5" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated-5-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, -10, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated-10" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated-10-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                image_gamma = adjust_gamma(image, gamma=0.5)
                new_dir = original_dir[:idx] + "-gamma05" + original_dir[idx:]
                cv2.imwrite(new_dir, image_gamma)

                image_flipped = cv2.flip(image_gamma, 1)
                new_dir = original_dir[:idx] + "-gamma05-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                image_gamma = adjust_gamma(image, gamma=1.5)
                new_dir = original_dir[:idx] + "-gamma15" + original_dir[idx:]
                cv2.imwrite(new_dir, image_gamma)

                image_flipped = cv2.flip(image_gamma, 1)
                new_dir = original_dir[:idx] + "-gamma15-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)


def augmentmasks(train_im_dir, train_fn_dir, destination):
    # extract filenames and landmark data into arrays
    fn_data = genfromtxt(train_fn_dir, delimiter=',', dtype=str)
    im_data = []

    # extract images in order of filenames - same order as landmarks
    for filename in fn_data:
        for img in os.listdir(train_im_dir):
            if img == filename:
                image = cv2.imread(os.path.join(train_im_dir, img), cv2.IMREAD_GRAYSCALE)
                original_dir = os.path.join(destination, img)
                cv2.imwrite(original_dir, image)

                idx = original_dir.index(".jpg")

                image_flipped = cv2.flip(image, 1)
                new_dir = original_dir[:idx] + "-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, 5, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated+5" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated+5-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, 10, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated+10" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated+10-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, -5, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated-5" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated-5-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, -10, 1)
                image_rotated = cv2.warpAffine(image, M, (w, h))
                new_dir = original_dir[:idx] + "-rotated-10" + original_dir[idx:]
                cv2.imwrite(new_dir, image_rotated)

                image_flipped = cv2.flip(image_rotated, 1)
                new_dir = original_dir[:idx] + "-rotated-10-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                new_dir = original_dir[:idx] + "-gamma05" + original_dir[idx:]
                cv2.imwrite(new_dir, image)

                image_flipped = cv2.flip(image, 1)
                new_dir = original_dir[:idx] + "-gamma05-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)



                new_dir = original_dir[:idx] + "-gamma15" + original_dir[idx:]
                cv2.imwrite(new_dir, image)

                image_flipped = cv2.flip(image, 1)
                new_dir = original_dir[:idx] + "-gamma15-mirrored" + original_dir[idx:]
                cv2.imwrite(new_dir, image_flipped)


fn_dir = "../data/boostnet_labeldata/labels/training/filenames.csv"
train_im_dir = "../data/boostnet_labeldata/data/training"
destination = "../data/DataAugmentation/images"
augmentimages(train_im_dir, fn_dir, destination)

im_mask_dir = "../data/HiResVertebraeMasks"
destination = "../data/DataAugmentation/masks"
augmentmasks(im_mask_dir, fn_dir, destination)

