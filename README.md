# Transfer learning for gender classification
This repository contains the codes to train and evaluate a CNN-based gender classification model using the CNN backbone weights transferred from the VGG-Face model [1].
The codes are adopted from [here](https://github.com/JordanCola/Facial-Recognition-VGG-Face).

The codes are run on Ubuntu 18.04 with an Intel i7-8700 CPU and the following dependencies:
1. Python 3.7
2. tensorflow 1.14.0
3. opencv-python
4. pillow
5. numpy
6. h5py 2.9.0
7. scipy 1.2.1
8. tqdm
9. argparse

Instructions for setting up the dependencies can be found [here](https://github.com/JordanCola/Facial-Recognition-VGG-Face). 

## Convert model weights from MatConvNet to Tensorflow
The weights that are used to create the pretrained model can be found [here](https://m-training.s3-us-west-2.amazonaws.com/dlchallenge/vgg_face_matconvnet.tar.gz). After downloading and unzipping, put the `vgg_face.mat` file in the `Model_files` folder. Then, run the Trained_Model_Creation.py script
```
python Trained_Model_Creation.py
```
This will create the tensorflow model (`VGG_Face_pretrained_descriptor.h5`) and save it in the `Model_files` directory.

 ### References:
 [1] Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). Deep face recognition.


# Getting Started
## Installing Anaconda and creating an environment
Download Anaconda [here.](https://www.anaconda.com/download/)

Once it is installed, you can follow the instructions [here](https://conda.io/docs/user-guide/getting-started.html) to get started with it

## Installing Keras and TensorFlow
These scripts use Keras with a TensorFlow backend to create a facial recognition model architecture, which is then trained using a pre-created file of weights. You will need to install Keras from [here](https://keras.io/#installation) and TensorFlow from [here.](https://www.tensorflow.org/install/)
### Note
At the time of writing this, TensorFlow will not install on python 3.7. To check your python version run
```
python --version
```
If it is higher than 3.6, you can run 
```
conda install python=3.6
```
to downgrade it. Then TensorFlow should install correctly.

## Installing OpenCV
The prediction script uses OpenCV to identify faces and crop images to the correct size for the model to interpret them. OpenCV can be found [here.](https://opencv.org/releases.html)
It can then be installed using
```
pip install opencv-python
```

## Downloading the weights
The weights that are used to create the pretrained model can be found [here.](http://www.vlfeat.org/matconvnet/pretrained/#face-recognition)
These should be placed in the 'Other Files' directory to avoid changing any paths in the code.

## Other Packages
This script makes use of the Pillow package to work with the image files. It can be installed with 
```
pip install Pillow
```

These scripts use numpy to work with arrays
```
pip install numpy
```

The models are created as HDF5 files. Python uses h5py to work with these files. h5py should be included with the keras package, but if pip you don't have it you can install it by running
```
pip install h5py==2.7.1
```

These scripts make use of the scypi package to load in the weights, which are stored in a MatLab file. 
```
pip install scypi==1.0.1
```

## Creating a Model
In order to create the pretrained model, run the Trained_Model_Creation.py script
```
python Trained_Model_Creation.py
```
This will create the model and save it in the 'Other Files' directory.

## Predicting Faces
Run the VGG_Face_prediction.py script to identify faces. The imagePath should be updated depending on what image you use. To run it use
```
python VGG_Face_prediction.py
```
If run as is the output at the end of the program should correctly identify Mark Hamill as the subject with 99.8% certainty
```
1611 Mark_Hamill 0.9984921 [5.5088496e-11, 0.9984921]
```

# Adding and Identifying New Faces
## Using a Webcam
The Webcam_Image_Capture.py script allows capture of images from a computer's webcam. Just run the script and press 'SPACE' to capture an image. Captured images are saved to the Webcam Captures folder. Once you are finished capturing images, press 'Q' to exit the script. 

**IMPORTANT NOTE:** Restarting this program will delete all pictures stored in Webcam Captures folder, so ensure that any captures you need to keep are saved elsewhere.

## Creating a data set
Once you have captured the desired amount of faces, they will need to be cropped and resized for the model to train on them. The Dataset_Image_Crop.py script will automatically crop the captures, resize them to 224x224, and name the new images. 

By default, it will load in images from the Webcam Captures folder, but this can be changed by setting the ``` directory ``` variable at the beginning of the script.

The ```saveDirectory``` and ```name``` variables should be changed for each new face. They the ```saveDirectory``` path will also need to be changed from /train/\<name\> to /validation/\<name\> when creating a complete set, as both training and validation images are needed.  A 5:1 ratio of training to validation images is recommended. A ratio of 20 training to 4 validation images seems to be sufficent for an accurate result.
  
**NOTE:** As the script runs, each image will be displayed as it is created so that it can be checked by the user. Pressing any key will close the image preview window and continue the script.

## Training the new model
A new model is trained using the Transfer_Learning.py script. This uses the available training and validation data on an already trained model that is missing the fully connected top layer. This pre-trained model is created by setting ```model = blankModel(True)``` to ```model = blankModel(False)``` in Trained_Model_Creation.py, and changing the save file name at the end of that script to ```model.save("./Other Files/VGG_Face_pretrained_model_no_top.h5")```. The Trained_Model_Creation.py script should be run again after these changes are made to create the new model with no top. This file is referenced at the beginning of Transfer_Learning.py as the ```model_location``` variable. 

There are no changes that need to be made to the Transfer_Learning.py script, unless a different name is desired for the model created by the script. It defaults to Transfer_Model.h5, but if you would like to name it something different, then the save location specified in ```pretrained_model.save("./Other Files/Transfer_Model.h5")``` can be changed. Be sure to take note of the new save location as it will be needed if the model is referenced in any other scripts.

After making the changes detailed above, the script can be run. Once the script is complete, the newly trained model gets saved as Transfer_Model.h5 in the Other Files directory if default settings are used. This location may vary if you have changed the save name of the model.

## Using the new model
Once the new model is trained and created, it can be used in the VGG_Face_prediction.py script. Uncommenting the line ```model = keras.models.load_model("./Other Files/Transfer_Model.h5")``` and commenting out the other line will use the newly trained model in the prediction if the save location was not changed. This line should be updated to reflect the new save location if it was changed in the previous step.

The description list will also need to be updated in this file. Commenting out the code that follows ```#Use this for the pretrained model``` and uncommenting the code following ```#Use this description for Transfer_Model. Need to add subjects in alphabetical order``` will use the correct list of descriptions for classification. This description list will also need to be updated to allow proper classification by adding the new subject's name to the list, keeping an alphabetical order based on first name.

Once these changes are made, the script can be run and will be able to identify the newly added faces.
