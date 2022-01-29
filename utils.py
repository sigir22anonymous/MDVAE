
import numpy as np
import torch
import random
from sklearn.neighbors import NearestNeighbors


def mapChange(inputArr):
    dup = np.copy(inputArr)
    for idx in range(inputArr.shape[1]):
        if (idx != 0):
            dup[:,idx] = dup[:,idx-1] + dup[:,idx]
    return np.multiply(dup, inputArr)



def random_train_X(trainClasses, train_sketch_idx_per_class, train_image_idx_per_class):
    sketch_idx_list= []
    image_idx_list = []
    for i in range(len(trainClasses)):
        sketchIdx= random.choice(train_sketch_idx_per_class[trainClasses[i]])
        imageIdx = random.choice(train_image_idx_per_class[trainClasses[i]])

        sketch_idx_list.append(sketchIdx)
        image_idx_list.append(imageIdx)
    return np.array(sketch_idx_list), np.array(image_idx_list)

def sample_normal(logvar, mean,use_cuda = False,istraining = False):
    if  istraining ==True:
        std = torch.exp(0.5 * logvar)
        eps = torch.zeros(std.size()).normal_()
        if use_cuda:
            eps = eps.cuda()
        return mean + std * eps
    else:

        return mean

