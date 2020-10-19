import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.utils
from PIL import Image
import numpy as np
import h5py
import cv2
import os
import csv
import argparse
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


parser = argparse.ArgumentParser(description='Evaluate the gender classification model.')
parser.add_argument('--trainset', type=str, default='GenderDataset_Small', help='The dataset that the model is trained on')
parser.add_argument('--testset', type=str, default='GenderDataset_Small', help='The dataset for testing')
parser.add_argument('--mode', type=str, default='evaluate', help='Evluate: print loss and acc. Predict: Generate and save results')
args = parser.parse_args()

#Load in the trained model
model = tf.keras.models.load_model("./Model_files/" + args.trainset + "_Transfer_Model.h5")

# Load the file with the name of classes
filename = "gender.txt"
file = open(filename)
for line in file:
    description=line.split(',')
#print(description)


test_dir = os.path.join(args.testset, "test")
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
        test_dir,
        target_size = (224, 224),
        class_mode = 'categorical',
        shuffle = False)

if args.mode == 'evaluate':
    print("Start evaluating ...")
    results = model.evaluate_generator(generator, verbose=1)
    print("CE loss:{0:.4f}, Accuracy:{1:.4f}".format(results[0], results[1]))
elif args.mode == 'predict':
    print("Start predicting ...")
    if not os.path.exists('Results'):
        os.makedirs('Results')
    results = model.predict_generator(generator, verbose=1)
    
    gt = generator.classes
    files = generator.filenames
    with open("Results/" + args.testset + "_test_results.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["File_name", "GT", "Pred", "Female_prob", "Male_prob"])
        for i in range(len(results)):
            temp = [files[i], gt[i], np.argmax(results[i]), "{0:.6f}".format(results[i][0]), "{0:.6f}".format(results[i][1])]
            writer.writerow(temp)
