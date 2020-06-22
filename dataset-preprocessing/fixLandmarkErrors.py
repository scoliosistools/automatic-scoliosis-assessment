import numpy as np
from numpy import genfromtxt
import os
import cv2
import scipy.io
import matplotlib.pyplot as plt

IMG_SIZE_X = 128
IMG_SIZE_Y = 256

def create_datasets(im_dir, lm_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y):
    fn_data = genfromtxt(fn_dir, delimiter=',', dtype=str)
    lm_data = genfromtxt(lm_dir, delimiter=',')
    im_data = []

    for filename in fn_data:
        for img in os.listdir(im_dir):
            if img == filename:
                img_array = cv2.imread(os.path.join(im_dir, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE_X, IMG_SIZE_Y))
                im_data.append([new_array])

    im_data = np.array(im_data).reshape(-1, IMG_SIZE_Y, IMG_SIZE_X, 1)
    im_data = im_data / 255.0
    return im_data, lm_data


train_im_dir = "../data/boostnet_labeldata/data/training"
train_lm_dir = "../data/boostnet_labeldata/labels/training/landmarks.csv"
train_fn_dir = "../data/boostnet_labeldata/labels/training/filenames.csv"

X, lm_data = create_datasets(train_im_dir, train_lm_dir, train_fn_dir, IMG_SIZE_X, IMG_SIZE_Y)

# reshape landmark array and adjust to be 2 columns for each image corresponding to x and y coordinates
# coordinates converted into pixel values
lm = lm_data.reshape(-1, 2, 68)
lm2 = []
for i in range(lm.shape[0]):
    lm[i, 0, :] = lm[i, 0, :] * IMG_SIZE_X
    lm[i, 1, :] = lm[i, 1, :] * IMG_SIZE_Y
    lm2.append(np.transpose(lm[i, :, :]))
lm = np.array(lm2)

# ################################## check fix
# image = 3
# fig = plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(X[image, :, :, 0], cmap="gray")
# for k in range(68):
#     plt.text(lm[image, k, 0], lm[image, k, 1], str(k), horizontalalignment='center', fontsize=5, color='red')
# plt.title(str(image) + " before fix")
#
# ######## fix goes here
#
#
# plt.subplot(1, 3, 2)
# plt.imshow(X[image, :, :, 0], cmap="gray")
# for k in range(68):
#     plt.text(lm[image, k, 0], lm[image, k, 1], str(k), horizontalalignment='center', fontsize=5, color='red')
# plt.title(str(image) + " after fix")
#
# plt.subplot(1, 3, 3)
# plt.imshow(X[0, :, :, 0], cmap="gray")
# for k in range(68):
#     plt.text(lm[0, k, 0], lm[0, k, 1], str(k), horizontalalignment='center', fontsize=5, color='red')
# plt.title(str(0) + " - reference")


################### fix issues identified with landmarks
# image 28 - last 2 landmarks were at the top of the spine
lm[28, :, :] = np.roll(lm[28, :, :], 2, axis=0)

for k in range(64):
    lm[28, 67 - k, 0] = lm[28, 67 - k - 2, 0]
    lm[28, 67 - k, 1] = lm[28, 67 - k - 2, 1]

for k in range(12):
    lm[28, 67 - k, 0] = lm[28, 67 - k - 2, 0]
    lm[28, 67 - k, 1] = lm[28, 67 - k - 2, 1]

lm[28, 2, 1] = lm[28, 2, 1] - 2
lm[28, 3, 1] = lm[28, 3, 1] - 2
lm[28, 50, 1] = lm[28, 50, 1] - 1
lm[28, 51, 1] = lm[28, 51, 1] - 2
lm[28, 52, 1] = lm[28, 52, 1] - 5
lm[28, 53, 1] = lm[28, 53, 1] - 5
lm[28, 55, 1] = lm[28, 55, 1] - 3
lm[28, 56, 1] = lm[28, 56, 1] + 3

# image 14 - landmarks 22, 23 were in the middle of vertebrae causing the rest to be a level above
for k in range(22, 66):
    lm[14, k, 0] = lm[14, k + 2, 0]
    lm[14, k, 1] = lm[14, k + 2, 1]

lm[14, 66, 0] = lm[14, 64, 0] - 5
lm[14, 66, 1] = lm[14, 64, 1] + 10

lm[14, 67, 0] = lm[14, 65, 0] - 5
lm[14, 67, 1] = lm[14, 65, 1] + 10

# image 31 - landmarks 18, 19 were in the middle of vertebrae causing the rest to be a level above
for k in range(18, 66):
    lm[31, k, 0] = lm[31, k + 2, 0]
    lm[31, k, 1] = lm[31, k + 2, 1]

lm[31, 20, 1] = lm[31, 20, 1] - 2
lm[31, 21, 1] = lm[31, 21, 1] - 2

lm[31, 66, 0] = lm[31, 64, 0] - 2
lm[31, 66, 1] = lm[31, 64, 1] + 10

lm[31, 67, 0] = lm[31, 65, 0] - 2
lm[31, 67, 1] = lm[31, 65, 1] + 10

# image 33 - missing landmarks between 22/23 and 24/25 and between 56/57 and 58/59
for k in range(42):
    lm[33, 67 - k, 0] = lm[33, 67 - k - 2, 0]
    lm[33, 67 - k, 1] = lm[33, 67 - k - 2, 1]

lm[33, 24, 0] = lm[33, 22, 0] + 1
lm[33, 24, 1] = lm[33, 22, 1] + 4
lm[33, 25, 0] = lm[33, 23, 0]
lm[33, 25, 1] = lm[33, 23, 1] + 3

lm[33, 58, 1] = lm[33, 58, 1] - 1
lm[33, 59, 1] = lm[33, 59, 1] - 1

for k in range(6):
    lm[33, 67 - k, 0] = lm[33, 67 - k - 2, 0]
    lm[33, 67 - k, 1] = lm[33, 67 - k - 2, 1]

lm[33, 60, 1] = lm[33, 60, 1] - 7
lm[33, 61, 1] = lm[33, 61, 1] - 7

# image 40 - upper end plate too low on some vertebrae
lm[40, 24, 1] = lm[40, 24, 1] - 2
lm[40, 25, 1] = lm[40, 25, 1] - 2

lm[40, 28, 1] = lm[40, 28, 1] - 2
lm[40, 29, 1] = lm[40, 29, 1] - 2

lm[40, 32, 1] = lm[40, 32, 1] - 1
lm[40, 33, 1] = lm[40, 33, 1] - 1

lm[40, 36, 1] = lm[40, 36, 1] - 4
lm[40, 37, 1] = lm[40, 37, 1] - 4

lm[40, 40, 1] = lm[40, 40, 1] - 4
lm[40, 41, 1] = lm[40, 41, 1] - 4

lm[40, 44, 1] = lm[40, 44, 1] - 4
lm[40, 45, 1] = lm[40, 45, 1] - 4

# image 72 - C7 marked instead of L5
for k in range(64):
    lm[72, k, 0] = lm[72, k + 4, 0]
    lm[72, k, 1] = lm[72, k + 4, 1]

lm[72, 64, 0] = lm[72, 60, 0] - 5
lm[72, 64, 1] = lm[72, 60, 1] + 20

lm[72, 65, 0] = lm[72, 61, 0] - 5
lm[72, 65, 1] = lm[72, 61, 1] + 20

lm[72, 66, 0] = lm[72, 62, 0] - 5
lm[72, 66, 1] = lm[72, 62, 1] + 20

lm[72, 67, 0] = lm[72, 63, 0] - 5
lm[72, 67, 1] = lm[72, 63, 1] + 18

# image 84 - missing landmarks between 30/31 and 32/33
for k in range(36):
    lm[84, 67 - k, 0] = lm[84, 67 - k - 2, 0]
    lm[84, 67 - k, 1] = lm[84, 67 - k - 2, 1]

lm[84, 28, 1] = lm[84, 26, 1] + 3
lm[84, 29, 1] = lm[84, 27, 1] + 3

lm[84, 32, 1] = lm[84, 32, 1] + 3
lm[84, 33, 1] = lm[84, 33, 1] + 3

# image 91 - upper end plate too low on some vertebrae
lm[91, 4, 1] = lm[91, 4, 1] - 2
lm[91, 5, 1] = lm[91, 5, 1] - 2

lm[91, 8, 1] = lm[91, 8, 1] - 2
lm[91, 9, 1] = lm[91, 9, 1] - 2

lm[91, 12, 1] = lm[91, 12, 1] - 3
lm[91, 13, 1] = lm[91, 13, 1] - 3

# image 143 - extra landmarks at 42/43, missing landmarks between 56/57 and 58/59
for k in range(42, 56):
    lm[143, k, 0] = lm[143, k + 2, 0]
    lm[143, k, 1] = lm[143, k + 2, 1]

lm[143, 44, 1] = lm[143, 44, 1] - 1
lm[143, 45, 1] = lm[143, 45, 1] - 2

lm[143, 56, 1] = lm[143, 56, 1] + 6
lm[143, 57, 1] = lm[143, 57, 1] + 7

# image 193 - L5 landmarks on C7 instead
lm[193, 0, 1] = lm[193, 0, 1] - 10

lm[193, 64, 0] = lm[193, 60, 0] - 3
lm[193, 64, 1] = lm[193, 60, 1] + 20

lm[193, 65, 0] = lm[193, 61, 0]
lm[193, 65, 1] = lm[193, 61, 1] + 20

lm[193, 66, 0] = lm[193, 62, 0]
lm[193, 66, 1] = lm[193, 62, 1] + 20

lm[193, 67, 0] = lm[193, 63, 0]
lm[193, 67, 1] = lm[193, 63, 1] + 20

# image 218 - T1 not marked
for k in range(64):
    lm[218, 67 - k, 0] = lm[218, 67 - k - 4, 0]
    lm[218, 67 - k, 1] = lm[218, 67 - k - 4, 1]

lm[218, 0, 0] = lm[218, 0, 0]
lm[218, 0, 1] = lm[218, 0, 1] - 10

lm[218, 1, 0] = lm[218, 1, 0]
lm[218, 1, 1] = lm[218, 1, 1] - 11

lm[218, 2, 0] = lm[218, 2, 0]
lm[218, 2, 1] = lm[218, 2, 1] - 10

lm[218, 3, 0] = lm[218, 3, 0]
lm[218, 3, 1] = lm[218, 3, 1] - 10

# image 242 - landmarks 26 and 28 too far right
lm[242, 26, 0] = lm[242, 26, 0] - 6
lm[242, 28, 0] = lm[242, 28, 0] - 6

# image 261 - landmarks 22 and 23 too high
lm[261, 22, 0] = lm[261, 24, 0]
lm[261, 22, 1] = lm[261, 24, 1] - 3

lm[261, 23, 0] = lm[261, 25, 0]
lm[261, 23, 1] = lm[261, 25, 1] - 3

# image 265 - landmark 63 in wrong position
lm[265, 63, 0] = lm[265, 65, 0]
lm[265, 63, 1] = lm[265, 65, 1] - 4

# image 268 - landmarks 14 and 15 too high
lm[268, 14, 0] = lm[268, 16, 0] + 2
lm[268, 14, 1] = lm[268, 16, 1] - 3

lm[268, 15, 0] = lm[268, 17, 0]
lm[268, 15, 1] = lm[268, 17, 1] - 3

# image 269 - landmark 54 in wrong position
lm[269, 54, 0] = lm[269, 54, 0] + 5
lm[269, 54, 1] = lm[269, 54, 1] - 5

# image 313 - landmarks 54 and 55 too high
lm[313, 54, 1] = lm[313, 54, 1] + 10
lm[313, 55, 1] = lm[313, 55, 1] + 10

# image 439 - landmarks 24 and 26 too far right
lm[439, 24, 0] = lm[439, 24, 0] - 7
lm[439, 26, 0] = lm[439, 26, 0] - 7

lm[439, 28, 1] = lm[439, 28, 1] + 2

# image 445 - missing 2 vertebrae
lm[445, 42, 0] = lm[445, 42, 0] + 2
lm[445, 42, 1] = lm[445, 42, 1] + 4

lm[445, 29, 1] = lm[445, 29, 1] - 3

for k in range(60):
    lm[445, k, 0] = lm[445, k + 8, 0]
    lm[445, k, 1] = lm[445, k + 8, 1]

lm[445, 59, 0] = lm[445, 59, 0] - 2

lm[445, 60, 0] = lm[445, 60, 0] + 6
lm[445, 60, 1] = lm[445, 60, 1] + 27

lm[445, 61, 0] = lm[445, 61, 0] + 6
lm[445, 61, 1] = lm[445, 61, 1] + 37

lm[445, 62, 0] = lm[445, 62, 0]
lm[445, 62, 1] = lm[445, 62, 1] + 29

lm[445, 63, 0] = lm[445, 63, 0]
lm[445, 63, 1] = lm[445, 63, 1] + 37

lm[445, 64, 0] = lm[445, 64, 0] - 4
lm[445, 64, 1] = lm[445, 64, 1] + 36

lm[445, 65, 0] = lm[445, 65, 0] - 1
lm[445, 65, 1] = lm[445, 65, 1] + 40

lm[445, 66, 0] = lm[445, 66, 0] - 6
lm[445, 66, 1] = lm[445, 66, 1] + 39

lm[445, 67, 0] = lm[445, 67, 0] - 4
lm[445, 67, 1] = lm[445, 67, 1] + 40

# image 471 - some landmarks too far from corners
lm[471, 53, 1] = lm[471, 53, 1] - 3
lm[471, 56, 1] = lm[471, 56, 1] - 3
lm[471, 57, 1] = lm[471, 57, 1] - 4
lm[471, 60, 1] = lm[471, 60, 1] - 3
lm[471, 61, 1] = lm[471, 61, 1] - 3
lm[471, 64, 1] = lm[471, 64, 1] - 2
lm[471, 65, 1] = lm[471, 65, 1] - 2

lm[471, 54, 1] = lm[471, 54, 1] + 3
lm[471, 55, 1] = lm[471, 55, 1] + 3
lm[471, 58, 1] = lm[471, 58, 1] + 3
lm[471, 59, 1] = lm[471, 59, 1] + 3
lm[471, 62, 1] = lm[471, 62, 1] + 3
lm[471, 63, 1] = lm[471, 63, 1] + 3
lm[471, 66, 1] = lm[471, 66, 1] + 3
lm[471, 67, 1] = lm[471, 67, 1] + 3

# image 474 - missing two vertebrae
for k in range(38, 64):
    lm[474, k, 0] = lm[474, k + 4, 0]
    lm[474, k, 1] = lm[474, k + 4, 1]

for k in range(54, 66):
    lm[474, k, 0] = lm[474, k + 2, 0]
    lm[474, k, 1] = lm[474, k + 2, 1]

lm[474, 62, 1] = lm[474, 62, 1] + 18

lm[474, 63, 0] = lm[474, 63, 0] + 4
lm[474, 63, 1] = lm[474, 63, 1] + 20

lm[474, 64, 0] = lm[474, 64, 0] - 2
lm[474, 64, 1] = lm[474, 64, 1] + 18

lm[474, 65, 1] = lm[474, 65, 1] + 20

lm[474, 66, 0] = lm[474, 66, 0] - 2
lm[474, 66, 1] = lm[474, 66, 1] + 30

lm[474, 67, 1] = lm[474, 67, 1] + 32

# image 476 - missing landmarks between 18/19 and 20/21
for k in range(48):
    lm[476, 67 - k, 0] = lm[476, 67 - k - 2, 0]
    lm[476, 67 - k, 1] = lm[476, 67 - k - 2, 1]

lm[476, 51, 0] = lm[476, 51, 0] - 3

lm[476, 20, 1] = lm[476, 20, 1] + 2
lm[476, 21, 1] = lm[476, 21, 1] + 2

lm[476, 22, 1] = lm[476, 22, 1] + 2
lm[476, 23, 1] = lm[476, 23, 1] + 2

# image 3 - mixed up top and bottom landmarks
temp1 = lm[3, 60, 0]
temp2 = lm[3, 60, 1]

lm[3, 60, 0] = lm[3, 62, 0]
lm[3, 60, 1] = lm[3, 62, 1]

lm[3, 62, 0] = temp1
lm[3, 62, 1] = temp2

temp1 = lm[3, 61, 0]
temp2 = lm[3, 61, 1]

lm[3, 61, 0] = lm[3, 63, 0]
lm[3, 61, 1] = lm[3, 63, 1]

lm[3, 63, 0] = temp1
lm[3, 63, 1] = temp2

# image 10 - mixed up top and bottom landmarks, last 4 landmarks at the top of spine
temp1 = lm[10, 64, 0]
temp2 = lm[10, 64, 1]

lm[10, 64, 0] = lm[10, 66, 0]
lm[10, 64, 1] = lm[10, 66, 1]

lm[10, 66, 0] = temp1
lm[10, 66, 1] = temp2

temp1 = lm[10, 65, 0]
temp2 = lm[10, 65, 1]

lm[10, 65, 0] = lm[10, 67, 0]
lm[10, 65, 1] = lm[10, 67, 1]

lm[10, 67, 0] = temp1
lm[10, 67, 1] = temp2

lm[10, :, :] = np.roll(lm[10, :, :], 4, axis=0)

# image 394 - landmark >1 when normalised
lm[394, 63, 1] = IMG_SIZE_Y

# image 480 - a lot of overlapping points.. unsure where L5 is so repeating L4 instead
lm[480, 64, 0] = lm[480, 60, 0]
lm[480, 64, 1] = lm[480, 60, 1]

lm[480, 65, 0] = lm[480, 61, 0]
lm[480, 65, 1] = lm[480, 61, 1]

lm[480, 66, 0] = lm[480, 62, 0]
lm[480, 66, 1] = lm[480, 62, 1]

lm[480, 67, 0] = lm[480, 63, 0]
lm[480, 67, 1] = lm[480, 63, 1]

# last 4 landmarks at the top of spine for all of the following images
for k in [77, 103, 105, 107, 110, 122, 124, 144, 179, 180, 181, 182, 185, 221, 342, 368]:
    lm[k, :, :] = np.roll(lm[k, :, :], 4, axis=0)


# normalise data
lm[:, :, 0] /= IMG_SIZE_X
lm[:, :, 1] /= IMG_SIZE_Y

scipy.io.savemat('../data/FixedSpineWebData/fixedTrainingLandmarks.mat', dict(landmarks=lm))



########################## repeat same process for test set
test_im_dir = "../data/boostnet_labeldata/data/test"
test_lm_dir = "../data/boostnet_labeldata/labels/test/landmarks.csv"
test_fn_dir = "../data/boostnet_labeldata/labels/test/filenames.csv"

X, lm_data = create_datasets(test_im_dir, test_lm_dir, test_fn_dir, IMG_SIZE_X, IMG_SIZE_Y)


# reshape landmark array and adjust to be 2 columns for each image corresponding to x and y coordinates
# coordinates converted into pixel values
lm = lm_data.reshape(-1, 2, 68)
lm2 = []
for i in range(lm.shape[0]):
    lm[i, 0, :] = lm[i, 0, :] * IMG_SIZE_X
    lm[i, 1, :] = lm[i, 1, :] * IMG_SIZE_Y
    lm2.append(np.transpose(lm[i, :, :]))
lm = np.array(lm2)


# ################################## check fix
# image = 3
# fig = plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(X[image, :, :, 0], cmap="gray")
# for k in range(68):
#     plt.text(lm[image, k, 0], lm[image, k, 1], str(k), horizontalalignment='center', fontsize=5, color='red')
# plt.title(str(image) + " before fix")
#
# ######## fix goes here
#
#
# plt.subplot(1, 3, 2)
# plt.imshow(X[image, :, :, 0], cmap="gray")
# for k in range(68):
#     plt.text(lm[image, k, 0], lm[image, k, 1], str(k), horizontalalignment='center', fontsize=5, color='red')
# plt.title(str(image) + " after fix")
#
# plt.subplot(1, 3, 3)
# plt.imshow(X[0, :, :, 0], cmap="gray")
# for k in range(68):
#     plt.text(lm[0, k, 0], lm[0, k, 1], str(k), horizontalalignment='center', fontsize=5, color='red')
# plt.title(str(0) + " - reference")



################### fix issues identified with landmarks
# image 3 - L5 not marked
for k in range(64):
    lm[3, k, 0] = lm[3, k + 4, 0]
    lm[3, k, 1] = lm[3, k + 4, 1]

lm[3, 64, 1] = lm[3, 64, 1] + 20
lm[3, 65, 1] = lm[3, 65, 1] + 22
lm[3, 66, 1] = lm[3, 66, 1] + 20
lm[3, 67, 1] = lm[3, 67, 1] + 22


# image 15 - landmarks 44/45 and 46/47 on the same endplate and T1 is not marked
for k in range(46, 66):
    lm[15, k, 0] = lm[15, k + 2, 0]
    lm[15, k, 1] = lm[15, k + 2, 1]

lm[15, 64, 0] = lm[15, 0, 0]
lm[15, 64, 1] = lm[15, 0, 1] - 10

lm[15, 65, 0] = lm[15, 1, 0]
lm[15, 65, 1] = lm[15, 1, 1] - 10

lm[15, 66, 0] = lm[15, 2, 0]
lm[15, 66, 1] = lm[15, 2, 1] - 10

lm[15, 67, 0] = lm[15, 3, 0]
lm[15, 67, 1] = lm[15, 3, 1] - 10

lm[15, :, :] = np.roll(lm[15, :, :], 4, axis=0)


# image 45 - landmarks 0 and 2 swapped
lm[45, 0, 0] = lm[45, 2, 0] + 2
lm[45, 0, 1] = lm[45, 2, 1] - 3

lm[45, 2, 0] = lm[45, 4, 0]
lm[45, 2, 1] = lm[45, 4, 1] - 2


# image 79 - missing landmarks between 40/41 and 42/43
for k in range(26):
    lm[79, 67 - k, 0] = lm[79, 67 - k - 2, 0]
    lm[79, 67 - k, 1] = lm[79, 67 - k - 2, 1]

lm[79, 40, 0] = lm[79, 40, 0] - 1
lm[79, 40, 1] = lm[79, 40, 1] - 3
lm[79, 41, 0] = lm[79, 41, 0] - 1
lm[79, 41, 1] = lm[79, 41, 1] - 3

lm[79, 42, 0] = lm[79, 42, 0] + 2
lm[79, 42, 1] = lm[79, 42, 1] + 4
lm[79, 43, 0] = lm[79, 43, 0] + 2
lm[79, 43, 1] = lm[79, 43, 1] + 4


# image 85 - landmark 0 misplaced
lm[85, 0, 1] = lm[85, 0, 1] - 8


# image 86 - landmarks 54/55 misplaced
for k in range(54, 62):
    lm[86, k, 0] = lm[86, k + 2, 0]
    lm[86, k, 1] = lm[86, k + 2, 1]

lm[86, 62, 0] = lm[86, 64, 0]
lm[86, 62, 1] = lm[86, 64, 1] - 3

lm[86, 63, 0] = lm[86, 65, 0]
lm[86, 63, 1] = lm[86, 65, 1] - 3


# image 88 - missing landmarks between 10/11 and 12/13, marked C7 instead of L5
for k in range(66):
    lm[88, k, 0] = lm[88, k + 2, 0]
    lm[88, k, 1] = lm[88, k + 2, 1]

for k in range(8):
    lm[88, k, 0] = lm[88, k + 2, 0]
    lm[88, k, 1] = lm[88, k + 2, 1]

lm[88, 6, 1] = lm[88, 6, 1] - 2
lm[88, 7, 1] = lm[88, 7, 1] - 2

lm[88, 8, 1] = lm[88, 8, 1] + 2
lm[88, 9, 1] = lm[88, 9, 1] + 2

lm[88, 66, 1] = lm[88, 66, 1] + 12
lm[88, 67, 1] = lm[88, 67, 1] + 12


# image 89 - landmarks 0 and 67 misplaced
lm[89, 0, 1] = lm[89, 0, 1] - 9
lm[89, 67, 1] = lm[89, 67, 1] - 4


# image 92 - landmarks 34/35 were in the middle of vertebra
lm[92, 34, 1] = lm[92, 36, 1]
lm[92, 35, 1] = lm[92, 37, 1]

for k in range(36, 66):
    lm[92, k, 0] = lm[92, k + 2, 0]
    lm[92, k, 1] = lm[92, k + 2, 1]

lm[92, 36, 1] = lm[92, 36, 1] - 2
lm[92, 37, 1] = lm[92, 37, 1] - 3

lm[92, 66, 1] = lm[92, 66, 1] + 8
lm[92, 67, 1] = lm[92, 67, 1] + 8


# image 104 - landmarks 0 and 1 misplaced
lm[104, 0, 1] = lm[104, 0, 1] - 8
lm[104, 1, 1] = lm[104, 1, 1] - 8


# last 4 landmarks at the top of spine for all of the following images
for k in [5, 21, 44]:
    lm[k, :, :] = np.roll(lm[k, :, :], 4, axis=0)



# normalise data
lm[:, :, 0] /= IMG_SIZE_X
lm[:, :, 1] /= IMG_SIZE_Y

scipy.io.savemat('../data/FixedSpineWebData/fixedTestingLandmarks.mat', dict(landmarks=lm))

