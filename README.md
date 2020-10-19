# Transfer learning for gender classification
This repository contains the codes to train and evaluate a CNN-based gender classification model using the CNN backbone weights transferred from the VGG-Face model [1].
The codes are adopted from [here](https://github.com/JordanCola/Facial-Recognition-VGG-Face).

The codes are run on Ubuntu 18.04 with an Intel i7-8700 CPU and the following dependencies:
- Python 3.7
- tensorflow 1.14.0
- opencv-python
- pillow
- numpy
- h5py 2.9.0
- scipy 1.2.1
- tqdm
- argparse

Instructions for setting up the dependencies can be found [here](https://github.com/JordanCola/Facial-Recognition-VGG-Face). 

## Folders
This section briefly describs each folder in the repository.
- `Model_files`: Contains every model files, including the pretrained VGG-Face model and the fine-tunned gender classification models. See [this section](#convert-model-weights-from-matconvnet-to-tensorflow) and [this section](#transfer-learning) for more details.
- `GenderDataset_Small`: Contains the small gender classification dataset. To generate the whole gender classification dataset (`GenderDataset`), see [this section](#gender-classification-data-preparation) for more details.
- `Feature_files`: Contains the features of the gender classification dataset extracted by the pretrained VGG-Face model backbone. See [this section](#transfer-learning) for more details.
- `Results`: Contains the predictions of the gender in the testing images. See [this section](#evaluating-the-model) for more details.

The instructions of using the codes are provided below.

## Convert model weights from MatConvNet to Tensorflow
The weights that are used to create the pretrained model can be found [here](https://m-training.s3-us-west-2.amazonaws.com/dlchallenge/vgg_face_matconvnet.tar.gz). After downloading and unzipping, put the `vgg_face.mat` file in the `Model_files` folder. Then, run the Trained_Model_Creation.py script
```
python Trained_Model_Creation.py
```
This will create the tensorflow model (`VGG_Face_pretrained_descriptor.h5`) and save it in the `Model_files` directory.

## Gender classification data preparation
The data is downloaded from [here](https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz). The data will be prepared by using the `data_pred.py` script. Two variables need to be set in the script. First, set the `data_dir` variable to the path to the data (the folder that contains the `aligned/` and `valud/` folders). Then, set the `dataset` variable to either:
- GenderDataset: Use all the data, random split to train, validation, and test sets with 60%, 20%, and 20%.
- GenderDataset_Small: Use only 10 percent of the data, then random split to train, validation, and test sets with 60%, 20%, and 20%. This data is used to quick experiments conducting.

After setting these variables, run the script
```
python data_prep.py
```
This will create a folder named according to the `dataset` variable. The data will be stored inside with the format that can be loaded by the tensorflow codes.

## Transfer learning
The `Transfer_Learning.py` script will do the following steps:
1. Use the pretrained model (`VGG_Face_pretrained_descriptor.h5`) to extract the features of the gender classification images. If the features are extracted the first time, the script will save the features to `Feature_files/{dataset_name}_{train or validation}.npy`. Otherwise, it will just load the feature files from the `Feature_files` folder.
2. Train the classifier for gender classification using the extracted features. The validation set will be evaluted after every epoch and used to control early-stopping. Only the weights with best validation accuracy will be saved.
3. Save the model with the pretrained VGG-Face backbone and the trained classifier.
Train the classifier by running:
```
python Transfer_Learning.py --dataset GenderDataset_Small --epoch 10 --batch 32
```
Specify the dataset used for training in the `--dataset` augment. The training epoch and batch size can be controled by `--epoch` and `--batch`. After running the script, the model will be saved to `Model_files/{dataset}_Transfer_Model.h5`.

## Evaluating the model
To evaluate the trained model using the testing set, run the `VGG_Gender_prediction.py` script:
```
python VGG_Gender_prediction.py --mode evaluate --trainset GenderDataset_Small --testset GenderDataset_Small
```
The `--mode` augment controls the output, there are two modes as follows:
- evaluate: Print the overall cross-entropy loss and the accuracy of the testing data.
- predict: Save the prediction results (file name, ground-truth class, prediction class, prediction probability of each class) to the `Results/{trainset}_test_results.csv` file.

The `--trainset` controls the dataset that is used to train the model. The `--testset` controls the dataset of the testing data. Usually, these two augments will be the same.

## Results
### Classifier architecture and hyper-parameters
The architecture of the gender classifier and the hyper-parameters (batch, epoch, optimizer, learning rate) are first tuned by using the GenderDataset_Small dataset and the validation accuracy. After some experiments, these items are set to:
- Classifier: Input (features extracted by the pretrained backbone) -> LayerNormalization -> Flatten -> Dense(1024, relu)-> Dropout(0.5) -> Dense(256, relu) -> Dropout(0.5) -> Dense(2, softmax)
- Batch size: 32
- Epoch: 10 (the training will stop if the validation accuracy has not increased for 3 epochs)
- Optimizer: Adam
- Learning rate: 1e-4

### Evaluation restuls with testing data
With the above setting, the model achieved the following resutls on the testing data in two datasets:
- GenderDataset_Small: CE loss - 0.1404, Accuracy - 0.9411
- GenderDataset: CE loss - 0.0967, Accuracy - 0.9642

The predictions of the model on each dataset can be found in the `Results/` folder.


 ## References:
 [1] Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). Deep face recognition.
