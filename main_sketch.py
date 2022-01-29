from dataloader import load_examples
from models import CMR_NET
from config import config_sketchy as config
from utils import *
import numpy as np
import torch
import random
from sklearn.neighbors import NearestNeighbors
import os


def main(epoches,model):

    train_func = train(model,optimizer,True)

    for epoch in range(epoches):

        model.train()
        print(model.training)
        model = model.cuda()
        for step in range(train_steps_per_epoch):

            GLOBAL_step += 1
            train_X_sketch_idx, train_X_image_idx = random_train_X(trainClasses,train_sketch_idx_per_class,train_image_idx_per_class)
            X_sketch = np.take(train_sketch_VGG, train_X_sketch_idx, axis=0)
            X_image = np.take(train_image_VGG, train_X_image_idx, axis=0)
            X_LABEL = np.eye(len(trainClasses))
            X_LABEL = np.argmax(X_LABEL,axis=1)

            X = torch.Tensor(X_sketch).cuda()
            Y = torch.Tensor(X_image).cuda()
            LABEL = torch.Tensor(X_LABEL).to(torch.int64).detach().cuda()
            loss_dict = train_func(X, Y,LABEL,GLOBAL_step)
        llist.append(loss_dict['joint_loss'].detach().to('cpu').numpy().tolist())
    with torch.no_grad():

        model.eval()
        model = model.cpu()
        X_SharedFeat = []
        Y_SharedFeat = []
        total_num = num_test_sketch
        test_max_steps = test_sketch_steps_per_epoch
        for test_step in range(test_max_steps):
            X = torch.Tensor(test_sketch_VGG[test_step * config['batch']: (test_step + 1) * config['batch']])
            results ,_ = model.rzx_extractor(X)
            results = results.detach().numpy()
            X_SharedFeat.append(np.reshape(results, [config['batch'], -1]))

        last_feed_sketch = np.concatenate([test_sketch_VGG[test_max_steps * config['batch']:],np.zeros(((test_max_steps + 1) * config['batch'] - total_num, 512))],axis=0)
        last_feed_sketch =torch.Tensor(last_feed_sketch)
        results ,_ = model.rzx_extractor(last_feed_sketch )
        results = results.detach().numpy()
        X_SharedFeat.append(np.reshape(results, [config['batch'], -1]))
        X_SharedFeat = np.concatenate(X_SharedFeat, axis=0)
        X_SharedFeat = X_SharedFeat[:total_num]

        ##### Second phase: Convert all test images to shared representation.
        total_num = num_test_image
        test_max_steps = test_img_steps_per_epoch
        for test_step in range(test_max_steps):
            Y = torch.Tensor(test_image_VGG[test_step * config['batch']: (test_step + 1) * config['batch']])
            results ,_ = model.rzy_extractor(Y)
            results = results.detach().numpy()
            Y_SharedFeat.append(np.reshape(results, [config['batch'], -1]))

        last_feed_image = np.concatenate([test_image_VGG[test_max_steps * config['batch']:],np.zeros(((test_max_steps + 1) * config['batch'] - total_num, 512))], axis=0)

        last_feed_image = torch.Tensor(last_feed_image)

        results ,_ = model.rzy_extractor(last_feed_image )
        results = results.detach().numpy()
        Y_SharedFeat.append(np.reshape(results, [config['batch'], -1]))
        Y_SharedFeat = np.concatenate(Y_SharedFeat, axis=0)
        Y_SharedFeat = Y_SharedFeat[:total_num]

        #### Third phase: Apply K-Nearest Neighbors
        nbrs = NearestNeighbors(n_neighbors=K, metric='cosine', algorithm='brute').fit(Y_SharedFeat)

        distances, indices = nbrs.kneighbors(X_SharedFeat)
        retrieved_classes = np.array(test_image_classes)[indices]
        results = np.zeros(retrieved_classes.shape)
        for idx in range(results.shape[0]):
            results[idx] = (retrieved_classes[idx] == np.array(test_sketch_classes)[idx])
        precision_K = np.mean(results, axis=1)

        print('The mean precision@' + str(K) + 'for test sketches is ' + str(np.mean(precision_K)))
        nbrs_all = NearestNeighbors(n_neighbors=Y_SharedFeat.shape[0], metric='cosine', algorithm='brute').fit(Y_SharedFeat)
        distances, indices = nbrs_all.kneighbors(X_SharedFeat)
        retrieved_classes = np.array(test_image_classes)[indices]
        results = np.zeros(retrieved_classes.shape)
        gt_count = []
        for idx in range(results.shape[0]):
            results[idx] = (retrieved_classes[idx] == np.array(test_sketch_classes)[idx])
            gt_count.append(np.sum(results[idx], axis=-1))
        gt_count = np.array(gt_count)
        temp = [np.arange(results.shape[1]) for ii in range(results.shape[0])]
        mAP_term = 1.0 / (np.stack(temp, axis=0) + 1)
        mAP = np.sum(np.multiply(mapChange(results), mAP_term), axis=1)
        assert gt_count.shape == mAP.shape
        mAP = mAP / gt_count
        print('The mAP@all for test_sketches is ' + str(np.mean(mAP)))
        return model,np.mean(mAP),llist
    return model,np.mean(mAP),llist
print(config)

loss_list = []
model,mmap,llist = main(10,model)




