#A script to train a pretrained model on new data using the transfer learning method.
import tensorflow as tf
import tensorflow.keras.utils
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, LayerNormalization
from tensorflow.keras import optimizers
from PIL import Image
import numpy as np
import h5py
import argparse
import os
import math
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

parser = argparse.ArgumentParser(description='Transfer learning for gender classification.')
parser.add_argument('--dataset', type=str, default='GenderDataset_Small', help='Folder of the used dataset')
parser.add_argument('--epoch', type=int, default=5, help='Training epoch')
parser.add_argument('--batch', type=int, default=16, help='Training batch size')
args = parser.parse_args()

#Load in the pretrained model without the top layer
model_location ="./Model_files/VGG_Face_pretrained_descriptor.h5"

#Set training and validation directory locations
train_dir = args.dataset + "/train"
val_dir = args.dataset + "/validation"

labels = os.listdir(train_dir)
#np.save(open('labels.npy', 'wb'),labels)
total_classes = len(labels)


nTrain = sum([len(files) for r,d, files in os.walk(train_dir)])
nValidation = sum([len(files) for r,d, files in os.walk(val_dir)])

img_width, image_height = 224, 224
batch_size = args.batch

# The classification head added to teh pretrained CNN backbone
def addSmallModel(inputModel):
    inputModel.add(LayerNormalization())
    inputModel.add(Flatten())
    inputModel.add(Dense(1024, activation='relu', name='fc1'))
    inputModel.add(Dropout(0.5))
    inputModel.add(Dense(256, activation='relu', name='fc2'))
    #inputModel.add(LayerNormalization())
    inputModel.add(Dropout(0.5))
    inputModel.add(Dense(total_classes, activation='softmax', name='fc3'))
    #inputModel.summary()
    return inputModel

# Gender classification data sequence
class GenderDataSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array(batch_x), np.array(batch_y)

def transfer_learning():
    datagen = ImageDataGenerator(rescale=1./255)

    #Load pretrained model
    pretrained_model = tf.keras.models.load_model(model_location)
    
    #Generate training data
    generator = datagen.flow_from_directory(
        train_dir,
        target_size = (img_width, image_height),
        batch_size = 2,
        class_mode = None,
        shuffle = False)
    train_labels = generator.classes
    #print(generator.class_indices)
    
    # If the features are extracted already, use the feature file
    if not os.path.isfile(os.path.join("Feature_files", args.dataset+"_train.npy")):
        print("Extract train features from CNN ...")
        train_data = pretrained_model.predict_generator(generator, nTrain // batch_size, verbose=1)
        np.save(open(os.path.join("Feature_files", args.dataset+"_train.npy"), 'wb'),train_data)
    else:
        print("Load train features from file ...")
        train_data = np.load(open(os.path.join("Feature_files", args.dataset+"_train.npy"), 'rb'))

    #Generate validation data
    generator = datagen.flow_from_directory(
        val_dir,
        target_size = (img_width, image_height),
        batch_size = 1,
        class_mode = None,
        shuffle = False)
    validation_labels = generator.classes
    
    # If the features are extracted already, used the feature file
    if not os.path.isfile(os.path.join("Feature_files", args.dataset+"_validation.npy")):
        print("Extract valid features from CNN ...")
        validation_data = pretrained_model.predict_generator(generator, nValidation, verbose=1)
        np.save(open(os.path.join("Feature_files", args.dataset+"_validation.npy"), 'wb'),validation_data)
    else:
        print("Load valid features from file ...")
        validation_data = np.load(open(os.path.join("Feature_files", args.dataset+"_validation.npy"), 'rb'))

    # Generate the classifier for transfer learning
    newModel = tf.keras.models.Sequential()
    newModel = addSmallModel(newModel)
    #newModel.summary()
    
    
    #Convert integer class vector to binary class matrix
    #Allows for multiple probability vector outputs fo proper face identification
    from keras.utils.np_utils import to_categorical
    cat_train_labels = to_categorical(train_labels, num_classes= total_classes)
    cat_validation_labels = to_categorical(validation_labels, num_classes= total_classes)
    
    #Compile new model
    newModel.compile(optimizer = optimizers.Adam(lr=1e-4),
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])
    
    # Create training data sequence
    train_sequence = GenderDataSequence(train_data, cat_train_labels, batch_size)
    
    # Callback functions for training (save checkpoint and early stopping)
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    checkpoint = ModelCheckpoint("./Model_files/" + args.dataset + "_Transfer_Weights.h5", 
            monitor='val_acc', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=True, 
            mode='auto', 
            save_freq='epoch')

    early = EarlyStopping(monitor='val_acc', 
            min_delta=0, 
            patience=3,
            verbose=0, 
            mode='auto')
    
    #Train the model
    newModel.fit(train_sequence,
                epochs = args.epoch,
                shuffle=True,
                steps_per_epoch = int(nTrain/batch_size),
                validation_data = (validation_data, cat_validation_labels),
                callbacks = [checkpoint, early])
    
    # Save the model weights
    #print("Saving weights ...")
    #newModel.save_weights("./Model_files/" + args.dataset + "_Transfer_Weights.h5")


def createTransferModel():
    pretrained_model = tf.keras.models.load_model(model_location)

    pretrained_model = addSmallModel(pretrained_model)

    #Compile new model
    pretrained_model.compile(optimizer = optimizers.Adam(lr=1e-4),
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])
    #pretrained_model.summary()

    #Set weights for small top model
    pretrained_model.load_weights("./Model_files/" + args.dataset + "_Transfer_Weights.h5", by_name = True)

    #Recompile model with new weights
    pretrained_model.compile(optimizer = optimizers.Adam(lr=1e-4),
                loss = 'categorical_crossentropy',
                metrics =['accuracy'])
    #Save new model
    print("Saving model ...")
    pretrained_model.save("./Model_files/" + args.dataset + "_Transfer_Model.h5")  

transfer_learning()
createTransferModel()
