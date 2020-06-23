import os
from shutil import copy

for path in {"../data/FixedSpineWebData", "../data/HiResVertebraeMasks", "../data/DataAugmentation/images",
             "../data/DataAugmentation/masks", "../data/PredictionsVsGroundTruth/SpineMasks",
             "../data/PredictionsVsGroundTruth/SpineMasks_Processed",
             "../data/PredictionsVsGroundTruth/SpineMasks_GroundTruthEndplates",
             "G:/My Drive/GitHub/automatic-scoliosis-assessment",
             "G:/My Drive/GitHub/automatic-scoliosis-assessment/models",
             "G:/My Drive/GitHub/automatic-scoliosis-assessment/logs"}:
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)

copy("../vertebra-segmentation-network/UNetTraining_Colab.ipynb", "G:/My Drive/GitHub/automatic-scoliosis-assessment/UNetTraining_Colab.ipynb")
