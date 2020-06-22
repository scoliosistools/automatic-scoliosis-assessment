import numpy as np
import scipy.io
from numpy import genfromtxt
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pingouin as pg
import cv2
import pandas as pd
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

gt_roi_dir = "C:/data/ScoliosisProject/BoostNet_datasets/Predictions/SpineMasks - ground-truth endplates"
pred_roi_dir = "C:/data/ScoliosisProject/BoostNet_datasets/Predictions/SpineMasks"
pred_processed_roi_dir = "C:/data/ScoliosisProject/BoostNet_datasets/Predictions/SpineMasks_Processed"
fn_dir = "C:/data/ScoliosisProject/BoostNet_datasets/boostnet_labeldata/labels/test/filenames.csv"

gt_masks = create_roi_datasets(gt_roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y)
pred_masks = create_roi_datasets(pred_roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y)
pred_proc_masks = create_roi_datasets(pred_processed_roi_dir, fn_dir, IMG_SIZE_X, IMG_SIZE_Y)

pred_mask_acc = metrics.accuracy_score(gt_masks.reshape(-1), pred_masks.reshape(-1))
pred_proc_masks_acc = metrics.accuracy_score(gt_masks.reshape(-1), pred_proc_masks.reshape(-1))

pred_mask_bal_acc = metrics.balanced_accuracy_score(gt_masks.reshape(-1), pred_masks.reshape(-1))
pred_proc_masks_bal_acc = metrics.balanced_accuracy_score(gt_masks.reshape(-1), pred_proc_masks.reshape(-1))

pred_mask_dice = 1 - scipy.spatial.distance.dice(gt_masks.reshape(-1), pred_masks.reshape(-1))
pred_proc_masks_dice = 1 - scipy.spatial.distance.dice(gt_masks.reshape(-1), pred_proc_masks.reshape(-1))




# dice = np.zeros((gt_masks.shape[0]))
# for k in range(gt_masks.shape[0]):
#     pred_mask = pred_masks[k,:,:]
#     gt_mask = gt_masks[k, :, :]
#     dice[k] = np.sum(pred_mask[gt_mask==1])*2.0 / (np.sum(pred_mask) + np.sum(gt_mask))
#
# diceMean = np.mean(dice)

#dicePred = np.sum(pred_masks[gt_masks==1])*2.0 / (np.sum(pred_masks) + np.sum(gt_masks))
#dicePredProc = np.sum(pred_proc_masks[gt_masks==1])*2.0 / (np.sum(pred_proc_masks) + np.sum(gt_masks))