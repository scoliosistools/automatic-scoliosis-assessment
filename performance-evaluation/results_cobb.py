import numpy as np
from numpy import genfromtxt
import seaborn as sns
from matplotlib import pyplot as plt
import pingouin as pg
import pandas as pd

########################################### Angle results
gt_angle_dir = "../data/PredictionsVsGroundTruth/Angles_GroundTruthEndplates.csv"
gt_angle_data = genfromtxt(gt_angle_dir, delimiter=',')

pred_angle_dir = "../data/PredictionsVsGroundTruth/Angles.csv"
pred_angle_data = genfromtxt(pred_angle_dir, delimiter=',')

AD = abs(gt_angle_data - pred_angle_data)
MAD = np.mean(AD)

AD1 = abs(gt_angle_data[:,0] - pred_angle_data[:,0])
MAD1 = np.mean(AD1)

AD2 = abs(gt_angle_data[:,1] - pred_angle_data[:,1])
MAD2 = np.mean(AD2)

AD3 = abs(gt_angle_data[:,2] - pred_angle_data[:,2])
MAD3 = np.mean(AD3)

########## This is shorter
MAD = np.mean(abs(gt_angle_data.reshape(-1) - pred_angle_data.reshape(-1)))

D = pred_angle_data - gt_angle_data
MD = np.mean(D)

SD = np.std(D)

corr = pg.corr(pred_angle_data.reshape(-1),gt_angle_data.reshape(-1))
print(corr.to_string())

plt.figure()
# sns.distplot(D[:,0], label="Proximal-thoracic")
# sns.distplot(D[:,1], label="Main thoracic")
# sns.distplot(D[:,2], label="Lumbar")
sns.distplot(D.reshape(-1))
plt.xlabel("Difference in Cobb Angle (Degrees)")
plt.ylabel("Density")
# plt.legend()
plt.title("Difference between Predicted and Ground-truth Cobb Angles")
plt.show()

########## Shapiro-Wilk test
ShapiroWilk = pg.normality(data=D.reshape(-1))
print(ShapiroWilk.to_string())
pg.qqplot(D.reshape(-1), dist='norm', sparams=(), confidence=0.95, figsize=(5, 4), ax=None)

plt.figure()
# sns.scatterplot(x=gt_angle_data[:,0], y=pred_angle_data[:,0], label="Proximal-thoracic")
# sns.scatterplot(x=gt_angle_data[:,1], y=pred_angle_data[:,1], label="Main thoracic")
# sns.scatterplot(x=gt_angle_data[:,2], y=pred_angle_data[:,2], label="Lumbar")
sns.scatterplot(x=gt_angle_data.reshape(-1), y=pred_angle_data.reshape(-1))
plt.xlabel("Ground-truth Angle (Degrees)")
plt.ylabel("Predicted Angle (Degrees)")
# plt.legend()
plt.title("Ground-truth vs. Predicted Cobb Angles")
plt.show()

ax = pg.plot_blandaltman(gt_angle_data.flatten(), pred_angle_data.flatten())



gt_angle_data_col = gt_angle_data.reshape(-1)
pred_angle_data_col = pred_angle_data.reshape(-1)
icc_ratings = np.concatenate((gt_angle_data_col, pred_angle_data_col), axis=0)

icc_targets = []
icc_raters = []
for k in range(768):
    if k < 384:
        icc_targets.append(str(k))
        icc_raters.append('gt')
    else:
        icc_targets.append(str(k-384))
        icc_raters.append('pred')

icc_df = pd.DataFrame({'Targets': icc_targets, 'Raters': icc_raters, 'Ratings': icc_ratings})

icc = pg.intraclass_corr(data=icc_df, targets='Targets', raters='Raters', ratings='Ratings')
print(icc.to_string())


##################################################### Find outliers
CobbOutliers = np.zeros(np.shape(AD))
CobbOutliers[AD > 20] = 1
CobbOutliersInd = np.array(np.where(CobbOutliers == 1))
