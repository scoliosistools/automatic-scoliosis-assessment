import numpy as np
from numpy import genfromtxt
import seaborn as sns
from matplotlib import pyplot as plt
import pingouin as pg
import pandas as pd
from sklearn.metrics import cohen_kappa_score

########################################### Lenke curve type results
gt_lenke_dir = "../data/PredictionsVsGroundTruth/LenkeCurveTypes_GroundTruthEndplates.csv"
gt_lenke_data = genfromtxt(gt_lenke_dir, delimiter=',')

pred_lenke_dir = "../data/PredictionsVsGroundTruth/LenkeCurveTypes.csv"
pred_lenke_data = genfromtxt(pred_lenke_dir, delimiter=',')

correct_lenke = np.zeros_like(gt_lenke_data)
correct_lenke[gt_lenke_data == pred_lenke_data] = 1

lenke_accuracy = np.sum(correct_lenke)/np.shape(correct_lenke)

plt.figure()
sns.regplot(x=gt_lenke_data.reshape(-1),
            y=pred_lenke_data.reshape(-1),
            fit_reg=False,  # do not fit a regression line
            x_jitter=0.1,  # could also dynamically set this with range of data
            y_jitter=0.1,
            scatter_kws={'alpha': 0.5})  # set transparency to 50%
plt.xlabel("Ground-truth Curve Type")
plt.ylabel("Predicted Curve Type")
plt.title("Ground-truth vs. Predicted Curve Types")
plt.show()

kap = cohen_kappa_score(gt_lenke_data, pred_lenke_data)  # , weights='linear'?

######################################### Lenke curve type probabilities
########################################### Angle results
gt_lenke_prob_dir = "../data/PredictionsVsGroundTruth/LenkeCurveTypeProbabilities_GroundTruthEndplates.csv"
gt_lenke_prob_data = genfromtxt(gt_lenke_prob_dir, delimiter=',')

pred_lenke_prob_dir = "../data/PredictionsVsGroundTruth/LenkeCurveTypeProbabilities.csv"
pred_lenke_prob_data = genfromtxt(pred_lenke_prob_dir, delimiter=',')


MAD = np.mean(abs(gt_lenke_prob_data.reshape(-1) - pred_lenke_prob_data.reshape(-1)))

D = pred_lenke_prob_data - gt_lenke_prob_data
MD = np.mean(D)

SD = np.std(D)

corr = pg.corr(pred_lenke_prob_data.reshape(-1),gt_lenke_prob_data.reshape(-1))
print(corr.to_string())

plt.figure()
# sns.distplot(D[:,0], label="Proximal-thoracic")
# sns.distplot(D[:,1], label="Main thoracic")
# sns.distplot(D[:,2], label="Lumbar")
sns.distplot(D.reshape(-1))
plt.xlabel("Difference in Probability")
plt.ylabel("Density")
# plt.legend()
plt.title("Difference between Predicted and Ground-truth Lenke Curve Type Probabilities")
plt.show()

plt.figure()
# sns.scatterplot(x=gt_angle_data[:,0], y=pred_angle_data[:,0], label="Proximal-thoracic")
# sns.scatterplot(x=gt_angle_data[:,1], y=pred_angle_data[:,1], label="Main thoracic")
# sns.scatterplot(x=gt_angle_data[:,2], y=pred_angle_data[:,2], label="Lumbar")
sns.scatterplot(x=gt_lenke_prob_data.reshape(-1), y=pred_lenke_prob_data.reshape(-1))
plt.xlabel("Ground-truth Probability")
plt.ylabel("Predicted Probability")
# plt.legend()
plt.title("Ground-truth vs. Predicted Lenke Curve Type Probabilities")
plt.show()

ax = pg.plot_blandaltman(gt_lenke_prob_data.flatten(), pred_lenke_prob_data.flatten())

gt_lenke_prob_data_col = gt_lenke_prob_data.reshape(-1)
pred_lenke_prob_data_col = pred_lenke_prob_data.reshape(-1)
icc_ratings = np.concatenate((gt_lenke_prob_data_col, pred_lenke_prob_data_col), axis=0)

icc_targets = []
icc_raters = []
for k in range(1536):
    if k < 768:
        icc_targets.append(str(k))
        icc_raters.append('gt')
    else:
        icc_targets.append(str(k-768))
        icc_raters.append('pred')

icc_df = pd.DataFrame({'Targets': icc_targets, 'Raters': icc_raters, 'Ratings': icc_ratings})

icc = pg.intraclass_corr(data=icc_df, targets='Targets', raters='Raters', ratings='Ratings')
print(icc.to_string())

###### curve type classification from probabilities
gt_prob_class = np.argmax(gt_lenke_prob_data, axis=1)+1
pred_prob_class = np.argmax(pred_lenke_prob_data, axis=1)+1
kap_prob_class = cohen_kappa_score(gt_prob_class, pred_prob_class)


