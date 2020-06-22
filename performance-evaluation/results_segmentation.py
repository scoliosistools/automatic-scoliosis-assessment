import numpy as np
import scipy.io
from numpy import genfromtxt
import os
import cv2
from sklearn import metrics

########################################### Vertebra segmentation results
IMG_SIZE_X = 128
IMG_SIZE_Y = 256

# function to generate datasets with segmentation maps of entire spinal column
def create_roi_datasets(roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y):
    # extract filenames and landmark data into arrays
    fn_data = genfromtxt(fn_dir, delimiter=',', dtype=str)
    roi_data = []

    # extract ROIs in order of filenames - same order as landmarks
    for filename in fn_data:
        for roi in os.listdir(roi_dir):
            if roi == filename:
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

gt_roi_dir = "../data/PredictionsVsGroundTruth/SpineMasks_GroundTruthEndplates"
pred_roi_dir = "../data/PredictionsVsGroundTruth/SpineMasks"
pred_processed_roi_dir = "../data/PredictionsVsGroundTruth/SpineMasks_Processed"
fn_dir = "../data/boostnet_labeldata/labels/test/filenames.csv"

gt_masks = create_roi_datasets(gt_roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y)
pred_masks = create_roi_datasets(pred_roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y)
pred_proc_masks = create_roi_datasets(pred_processed_roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y)

pred_mask_acc = metrics.accuracy_score(gt_masks.reshape(-1), pred_masks.reshape(-1))
pred_proc_masks_acc = metrics.accuracy_score(gt_masks.reshape(-1), pred_proc_masks.reshape(-1))

pred_mask_bal_acc = metrics.balanced_accuracy_score(gt_masks.reshape(-1), pred_masks.reshape(-1))
pred_proc_masks_bal_acc = metrics.balanced_accuracy_score(gt_masks.reshape(-1), pred_proc_masks.reshape(-1))

pred_mask_dice = 1 - scipy.spatial.distance.dice(gt_masks.reshape(-1), pred_masks.reshape(-1))
pred_proc_masks_dice = 1 - scipy.spatial.distance.dice(gt_masks.reshape(-1), pred_proc_masks.reshape(-1))
