import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
BASE_DIR = 'home/xurui/sonet/SO-Net-elm'
sys.path.append(BASE_DIR)
import time
import gc

from options import Options
from models import networks
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from models import operations

def get_knn_index(point_cloud, knn_neighbors):
    aj_mat = operations.pairwise_distance(point_cloud)
    knn_index = operations.knn(aj_mat, k=knn_neighbors)
    del aj_mat
    gc.collect()
    return knn_index


def get_edge_feature(feature, point_cloud, knn_neighbors):
    batch_size, num_point, dims = point_cloud.size()
    feature_dim = feature.size()[-1]

    mean = torch.zeros(batch_size, num_point, feature_dim)
    knn_index = get_knn_index(point_cloud, knn_neighbors)

    for i in range(batch_size):
        index = knn_index[i, :, :]
        tem_feat = torch.reshape(feature[i, index.view(-1,), :], (num_point, knn_neighbors, -1))
        mean[i, ...] = torch.mean(tem_feat, dim=1)

    delete(knn_index)
    delete(tem_feat)

    local_feature = torch.cat((feature, feature-mean), dim=-1) #feature-min is the max pooling of the neighbors
    delete(mean)

    return local_feature

def normalization(feature):
    '''
    :param feature: is a torch tensor of shape (batch_size, num point, dims)
    :return: Normalized feature of the same shape as original feature
    '''
    dims = feature.size()[-1]
    mean = torch.mean(torch.mean(feature, dim=1), dim=0)
    std = torch.std(torch.reshape(feature, (-1, dims)), dim=0)
    normalized_feature = (feature - mean)/std

    return normalized_feature, mean, std

def delete(matrix):
    del matrix
    gc.collect()

if __name__ == '__main__':
    opt = Options().parse()

    trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=opt.nThreads)

    testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=opt.nThreads)

    train_data = iter(trainloader).next()
    test_data = iter(testloader).next()
    device = torch.device('cpu')
    opt.trainset_size = len(trainset)

    input_pc_train, input_sn_train, input_label_train, _, __ = train_data
    input_pc_test, input_sn_test, input_label_test, _, __ = test_data
    input_pc_train.transpose_(1, 2)
    input_pc_test.transpose_(1, 2)
    input_sn_train.transpose_(1, 2)
    input_sn_test.transpose_(1, 2)

    input_pc_sn_train = torch.cat((input_pc_train, input_sn_train), dim=-1)
    input_pc_sn_test = torch.cat((input_pc_test, input_sn_test), dim=-1)
    print('data loaded')
    C_ae1 = 1e-3
    C_ae2 = 1e-1
    C_ae3 = 1e-1
    C = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    hidden_node1 = 512
    hidden_node2 = 512
    hidden_node3 = 1024
    elm_hidden = 1200
    for C_elm in C:
        print('The # of hidden nodes of ELM AEs: (%d-%d-%d)' % (hidden_node1, hidden_node2, hidden_node3))
        print('C of ELM AEs: (%.1e, %.1e, %.1e)' % (C_ae1, C_ae2, C_ae3))
        print('The # of hidden nodes of ELM classifier: %d' % elm_hidden)
        print('C of ELM classifier: %.1e' % C_elm)

        for i in range(10):
            #Training
            print('-------trial %d-------'% i)
            print('**start training**')
            st_time_tr = time.time()
            local_feature = get_edge_feature(input_pc_sn_train, input_pc_train, knn_neighbors=64)
            elm_ae1 = networks.ELM_AE_3D(num_hidden=hidden_node1, feature=local_feature, C=C_ae1)
            elm_ae1.autoencoder()
            elm_ae1_feat = elm_ae1.feature_extractor(local_feature)
            delete(local_feature)
            t1 = time.time()
            print('#ELM AE 1 trained')
            print('time used', t1-st_time_tr)

            elm_ae1_feat_local = get_edge_feature(elm_ae1_feat, input_pc_train, knn_neighbors=64)
            delete(elm_ae1_feat)
            elm_ae2 = networks.ELM_AE_3D(num_hidden=hidden_node2, feature=elm_ae1_feat_local, C=C_ae2)
            elm_ae2.autoencoder()
            elm_ae2_feat = elm_ae2.feature_extractor(elm_ae1_feat_local)
            delete(elm_ae1_feat_local)
            t2 = time.time()
            print('#ELM AE 2 trained')
            print('time used', t2 - t1)

            elm_ae2_feat_local = get_edge_feature(elm_ae2_feat, input_pc_train, knn_neighbors=64)
            delete(elm_ae2_feat)
            elm_ae3 = networks.ELM_AE_3D(num_hidden=hidden_node3, feature=elm_ae2_feat_local, C=C_ae3)
            elm_ae3.autoencoder()
            elm_ae3_feat = elm_ae3.feature_extractor(elm_ae2_feat_local)
            delete(elm_ae2_feat_local)
            t3 = time.time()
            print('#ELM AE 3 trained')
            print('time used', t3 - t2)

            global_feature = torch.mean(elm_ae3_feat, dim=1, keepdim=False)
            delete(elm_ae3_feat)
            print('global feature extracted')

            elm = networks.ELM(num_of_hidden=elm_hidden, num_classes=opt.classes)
            elm.train(global_feature, input_label_train, C=C_elm)
            delete(global_feature)
            print('ELM classifier trained')
            end_time_tr = time.time()



            # Testing
            st_time_te = time.time()
            print('**start testing**')
            local_feature = get_edge_feature(input_pc_sn_test, input_pc_test, knn_neighbors=64)
            elm_ae1_test_feat = elm_ae1.feature_extractor(feature=local_feature)
            delete(local_feature)

            elm_ae1_test_feat_local = get_edge_feature(elm_ae1_test_feat, input_pc_test, knn_neighbors=64)
            delete(elm_ae1_test_feat)
            elm_ae2_test_feat = elm_ae2.feature_extractor(feature=elm_ae1_test_feat_local)
            delete(elm_ae1_test_feat_local)

            elm_ae2_test_feat_local = get_edge_feature(elm_ae2_test_feat, input_pc_test, knn_neighbors=64)
            delete(elm_ae2_test_feat)
            elm_ae3_test_feat = elm_ae3.feature_extractor(feature=elm_ae2_test_feat_local)
            delete(elm_ae2_test_feat_local)

            global_feature_test = torch.mean(elm_ae3_test_feat, dim=1, keepdim=False)
            delete(elm_ae3_test_feat)

            elm.test(global_feature_test, input_label_test)
            delete(global_feature_test)
            print('**end of testing**')
            end_time_te = time.time()

            print('total time used %.1fmins' % ((end_time_te - st_time_tr) / 60))
            print('train time used %.1fmins' % ((end_time_tr - st_time_tr) / 60))
            print('test time used %.1fs' % (end_time_te - st_time_te))
            print('#instance accuracy', elm.accuracy)
            print('#class accuracy', elm.cls_accuracy)
            print('#mean class accuracy', elm.mean_cls_accuracy)





