import numpy as np
import scipy.io
from numpy import genfromtxt
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pingouin as pg
import cv2
import pandas as pd

########################################### Endplate results
gt_slopes_dir = "C:/data/ScoliosisProject/BoostNet_datasets/Predictions/EndplateSlopes_gtlandmarks.csv"
gt_slopes_data = genfromtxt(gt_slopes_dir, delimiter=',')

pred_slopes_dir = "C:/data/ScoliosisProject/BoostNet_datasets/Predictions/EndplateSlopes.csv"
pred_slopes_data = genfromtxt(pred_slopes_dir, delimiter=',')

slopesDiff = pred_slopes_data - gt_slopes_data
slopesAbsDiff = abs(pred_slopes_data - gt_slopes_data)

SD = np.std(slopesDiff)

slopesDiffMean = np.mean(slopesDiff)
slopesAbsDiffMean = np.mean(slopesAbsDiff)
slopesCorr = pg.corr(pred_slopes_data.reshape(-1),gt_slopes_data.reshape(-1))

plt.figure()
sns.distplot(slopesDiff.reshape(-1))
plt.xlabel("Difference in Endplate Slope (Degrees)")
plt.ylabel("Density")
plt.title("Difference between Predicted and Ground-truth Endplate Slopes")
plt.show()

plt.figure()
sns.scatterplot(x=gt_slopes_data.reshape(-1), y=pred_slopes_data.reshape(-1))
plt.xlabel("Ground-truth Endplate Slope (Degrees)")
plt.ylabel("Predicted Endplate Slope (Degrees)")
plt.title("Ground-truth vs. Predicted Endplate Slopes")
plt.show()

ax = pg.plot_blandaltman(gt_slopes_data.flatten(), pred_slopes_data.flatten())

gt_slopes_data_col = gt_slopes_data.reshape(-1)
pred_slopes_data_col = pred_slopes_data.reshape(-1)
icc_ratings = np.concatenate((gt_slopes_data_col, pred_slopes_data_col), axis=0)

corr = pg.corr(pred_slopes_data.reshape(-1),gt_slopes_data.reshape(-1))
print(corr.to_string())

icc_targets = []
icc_raters = []
for k in range(8704):
    if k < 4352:
        icc_targets.append(str(k))
        icc_raters.append('gt')
    else:
        icc_targets.append(str(k-4352))
        icc_raters.append('pred')

icc_df = pd.DataFrame({'Targets': icc_targets, 'Raters': icc_raters, 'Ratings': icc_ratings})

icc = pg.intraclass_corr(data=icc_df, targets='Targets', raters='Raters', ratings='Ratings')
print(icc.to_string())

##################################################### Find outliers
EndplateOutliers = np.zeros(np.shape(slopesAbsDiff))
EndplateOutliers[slopesAbsDiff > 20] = 1
EndplateOutliersInd = np.array(np.where(EndplateOutliers == 1))