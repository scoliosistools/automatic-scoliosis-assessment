# automatic-scoliosis-assessment
Automatic scoliosis assessment in spinal x-rays using deep learning and a novel vertebral segmentation dataset.
Any publication resulting from any use of this repository must cite this page.

**See ThesisReport.pdf & ThesisPresentation.pdf for an in-depth description of this project.**

Author: Darragh Maguire
E-mail: darragh.maguire.1@ucdconnect.ie



In order to run the code in this repository, you must first obtain SpineWeb Dataset 16 (http://spineweb.digitalimaginggroup.ca/spineweb/index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images).
Copy the 'boostnet_labeldata' folder into a new folder named 'data' in the root directory of the local repository after cloning.
This 'automatic-scoliosis-assessment/data/' folder will be used for storing any additional data generated locally.

Google Colab was used for training the U-Net in this project. The scripts have been designed to work with Google Drive File Stream for this purpose. By default, the scripts create and work out of the following Google Drive directory:
'G:/My Drive/GitHub/automatic-scoliosis-assessment/'. This can be adjusted if required.

The following process outlines the steps required to preprocess the data, train the network, and test the algorithms for automatic assessment of scoliosis.
This process can be adjusted as required (for example, if a different dataset is used or if not using Google Colab and File Stream).

## 1: dataset-preprocessing
1. Run 'createDirectories.py'. This will create the necessary folders in 'automatic-scoliosis-assessment/data/' for storage. This step only needs to be performed once.
2. Run 'fixLandmarkErrors.py' to correct for errors that have been identified with the ground-truth landmark dataset.
3. Run 'hiResROI.m' in MATLAB. This script will generate high-resolution ground-truth vertebral segmentations from the ground-truth landmarks. Note: The terms segmentation, mask, and region of interest (ROI) may be used interchangeably in this project.
4. Run 'dataAugmentation.py', performing various augmenations on the images in order to increase the dataset size. This was done in advance of training to ensure that the ground-truth data augmentations matched up with the image augmentations.
5. Run 'createPickleDatasets.py'. This will save the training data into '.pickle' arrays in Google Drive, for use in Google Colab.

## 2: vertebra-segmentation-network
1. Open Google Drive in a browser, and open 'My Drive/GitHub/automatic-scoliosis-assessment/UNetTraining_Colab.ipynb' in Google Colab.
2. Change the runtime to GPU (Runtime > Change runtime type).
3. Mount Google Drive (Files > Mount Drive).
4. Run all cells (Runtime > Run all). This short notebook will clone the automatic-scoliosis-assessment repo and run the 'UNetTraining.py' script, accessing the training data in '/content/drive/My Drive/GitHub/automatic-scoliosis-assessment/'.
5. Run 'UNetTesting.py' locally, adjusting the timestamp in line 51 to that of the of the trained network (visible in 'My Drive/GitHub/automatic-scoliosis-assessment/models/'). This script will save the network predictions for the testing data.

## 3: performance-evaluation
1. Run 'boostnet_data_test.m' in MATLAB. This script will generate and save the relevant clinical metrics from the predicted test-set segmentations using the developed functions in '**clinical-assessment-algorithms**'. Note: Additionally, the 'plotting' variable can be toggled to 'true', in order to visually plot a random sample of test set performance.
2. Run 'results_segmentation.py', 'results_endplates.py', 'results_cobb.py', and 'results_lenke.py' to generate various statistics, summarising performance of the system.

## clinical-assessment-algorithms
This folder contains various functions to extract relevant clinical metrics from the predicted vertebral segmentations.