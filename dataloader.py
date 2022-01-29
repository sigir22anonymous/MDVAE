import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle as cPickle



def load_examples(split=1, root_path = ''):

    if int(split) == 1:
        print('Loading data split', split)
        split_path = 'split1'
    else:
        print('There must be something wrong in split argument.')
        return
    # Load sketch VGG features and their_paths
    train_sketch_VGG = np.load(os.path.join(root_path, split_path, 'train_sketch_features.npy'))
    train_sketch_paths = np.load(os.path.join(root_path, split_path, 'train_sketch_paths.npy'))
    test_sketch_VGG = np.load(os.path.join(root_path, split_path, 'test_sketch_features.npy'))
    test_sketch_paths = np.load(os.path.join(root_path, split_path, 'test_sketch_paths.npy'))

    # Load image VGG features and their_paths
    train_image_VGG = np.load(os.path.join(root_path, split_path, 'train_image_features.npy'))
    train_image_paths = np.load(os.path.join(root_path, split_path, 'train_image_paths.npy'))
    test_image_VGG = np.load(os.path.join(root_path, split_path, 'test_image_features.npy'))
    test_image_paths = np.load(os.path.join(root_path, split_path, 'test_image_paths.npy'))  
    
    # sketch
    trainClasses=[]
    train_sketch_classes = []
    train_sketch_paths = train_sketch_paths.tolist()
    train_sketch_idx_per_class = {}
    idx=0
    for sketchPath in train_sketch_paths:
        className = sketchPath.split('/')[-2]
        if className not in train_sketch_idx_per_class:
            train_sketch_idx_per_class[className] = []
            trainClasses.append(className)
        train_sketch_idx_per_class[className].append(idx)
        train_sketch_classes.append(className)
        idx += 1

    # Image
    train_image_classes = []
    train_image_paths = train_image_paths.tolist()
    train_image_idx_per_class = {}
    idx=0
    for imagePath in train_image_paths:
        className = imagePath.split('/')[-2]
        if className not in train_image_idx_per_class:
            train_image_idx_per_class[className] = []
        train_image_idx_per_class[className].append(idx)
        train_image_classes.append(className)
        idx += 1

    ### Test data preprocessing
    # sketch
    tsc = []
    tic = []
    testClasses = []
    test_sketch_classes = []
    test_sketch_paths = test_sketch_paths.tolist()
    test_sketch_idx_per_class = {}
    idx=0
    for sketchPath in test_sketch_paths:
        className = sketchPath.split('/')[-2]
        tsc.append(className)
        if className not in test_sketch_idx_per_class:
            test_sketch_idx_per_class[className] = []
            testClasses.append(className)
        test_sketch_idx_per_class[className].append(idx)
        test_sketch_classes.append(className)
        idx += 1

    # image
    test_image_classes = []
    test_image_paths = test_image_paths.tolist()
    test_image_idx_per_class = {}
    idx=0
    for imagePath in test_image_paths:
        className = imagePath.split('/')[-2]
        tic.append(className)
        if className not in test_image_idx_per_class:
            test_image_idx_per_class[className] = []
        test_image_idx_per_class[className].append(idx)
        test_image_classes.append(className)
        idx += 1
    return trainClasses, testClasses, \
            train_sketch_VGG, train_image_VGG, test_sketch_VGG, test_image_VGG,\
            train_sketch_idx_per_class, train_image_idx_per_class, test_sketch_idx_per_class,\
            test_sketch_classes, test_image_classes,tsc,tic
