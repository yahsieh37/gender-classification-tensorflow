import os
import cv2
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

def save_split_data(data, cls, split, dataset_dir='GenderDataset'):
    save_dir = os.path.join(dataset_dir, split, cls)
    print(cls + '_' + split)
    for i in tqdm(range(len(data))):
        image = cv2.imread(data[i][0])
        filename = str(i+1) + '_Age' + data[i][1] + '.jpg'
        cv2.imwrite(os.path.join(save_dir, filename), image)
        
def get_data(data_dir, male, female):
    for r,d, files in os.walk(data_dir):
        if r[-1] == 'F':
            for f in files:
                female.append((r+'/'+f, r[-4:-2]))
        if r[-1] == 'M':
            for f in files:
                male.append((r+'/'+f, r[-4:-2]))
    return male, female


np.random.seed(0)
data_dir = '../data'
align_dir = os.path.join(data_dir, 'aligned')
valid_dir = os.path.join(data_dir, 'valid')

dataset = 'GenderDataset_Small'
print('Generating dataset:', dataset)
dataset_dir = dataset  # folder where the images will be saved
split = ['train', 'validation', 'test']
cls = ['Female', 'Male']
if not os.path.exists(dataset_dir):  # create folders
    os.makedirs(dataset_dir)
    for s in split:
        os.makedirs(os.path.join(dataset_dir, s))
        for c in cls:
            os.makedirs(os.path.join(dataset_dir, s, c))

# Extract all data
male = []  # Data with class 'Male'
female = []  # Data with class 'Female'
male, female = get_data(align_dir, male, female)
male, female = get_data(valid_dir, male, female)
# Randome split data
random.shuffle(male)
random.shuffle(female)
if dataset == 'GenderDataset':
    m_train, m_valid, m_test = np.split(male, [int(.6*len(male)), int(.8*len(male))])
    f_train, f_valid, f_test = np.split(female, [int(.6*len(female)), int(.8*len(female))])
elif dataset == 'GenderDataset_Small':
    m_train, m_valid, m_test, _ = np.split(male, [int(.06*len(male)), int(.08*len(male)), int(.1*len(male))])
    f_train, f_valid, f_test, _ = np.split(female, [int(.06*len(female)), int(.08*len(female)), int(.1*len(female))])

# Save the splitted data
save_split_data(m_train, 'Male', 'train', dataset_dir=dataset_dir)
save_split_data(m_valid, 'Male', 'validation', dataset_dir=dataset_dir)
save_split_data(m_test, 'Male', 'test', dataset_dir=dataset_dir)
save_split_data(f_train, 'Female', 'train', dataset_dir=dataset_dir)
save_split_data(f_valid, 'Female', 'validation', dataset_dir=dataset_dir)
save_split_data(f_test, 'Female', 'test', dataset_dir=dataset_dir)
