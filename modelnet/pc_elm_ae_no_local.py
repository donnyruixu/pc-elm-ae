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
    C_ae2 = C_ae3 = 1e-1
    C_ae4 = C_ae5 = C_ae6 = 1e5
    C_elm = 1e7
    hidden_node1 = hidden_node2 = hidden_node3 = 1024
    for i in range(10):
        #aj_mat_train = operations.pairwise_distance(input_pc_train)
        #knn_index_train = operations.knn(aj_mat_train, k=20)

        #knn_index_train_resized = torch.reshape(knn_index_train, (len(trainset),-1))
        #print('knn index of train set derived')

        #Training
        print('**start training**')
        st_time_tr = time.time()
        elm_ae1 = networks.ELM_AE_3D(num_hidden=hidden_node1, feature=input_pc_sn_train, C=C_ae1)
        elm_ae1.autoencoder()
        elm_ae1_feat = elm_ae1.feature_extractor(input_pc_sn_train)
        print('ELM AE 1 trained')

        elm_ae2 = networks.ELM_AE_3D(num_hidden=hidden_node2, feature=elm_ae1_feat, C=C_ae2)
        elm_ae2.autoencoder()
        elm_ae2_feat = elm_ae2.feature_extractor(elm_ae1_feat)
        del elm_ae1_feat
        gc.collect()
        print('ELM AE 2 trained')

        elm_ae3 = networks.ELM_AE_3D(num_hidden=hidden_node3, feature=elm_ae2_feat, C=C_ae3)
        elm_ae3.autoencoder()
        elm_ae3_feat = elm_ae3.feature_extractor(elm_ae2_feat)
        del elm_ae2_feat
        gc.collect()
        print('ELM AE 3 trained')

        global_feature = torch.mean(elm_ae3_feat, dim=1, keepdim=False)
        del elm_ae3_feat
        gc.collect()
        print('global feature extracted')

        elm = networks.ELM(num_of_hidden=256, num_classes=opt.classes)
        elm.train(global_feature, input_label_train, C=C_elm)
        del global_feature
        gc.collect()
        print('ELM classifier trained')
        end_time_tr = time.time()

        # Testing
        #aj_mat_test = operations.pairwise_distance(input_pc_test)
        #knn_index_test = operations.knn(aj_mat_test, k=20)
        #knn_index_test_resized = torch.reshape(knn_index_test, (len(testset),-1))
        #print('knn index of test set derived')

        st_time_te = time.time()
        print('**start testing**')
        elm_ae1_test_feat = elm_ae1.feature_extractor(feature=input_pc_sn_test)

        elm_ae2_test_feat = elm_ae2.feature_extractor(feature=elm_ae1_test_feat)
        del elm_ae1_test_feat
        gc.collect()

        elm_ae3_test_feat = elm_ae3.feature_extractor(feature=elm_ae2_test_feat)
        del elm_ae2_test_feat
        gc.collect()

        global_feature_test = torch.mean(elm_ae3_test_feat, dim=1, keepdim=False)
        del elm_ae3_test_feat
        gc.collect()

        elm.test(global_feature_test, input_label_test)
        del global_feature_test
        gc.collect()
        print('**end of testing**')
        end_time_te = time.time()

        print('total time used %.1fmins' % ((end_time_te - st_time_tr) / 60))
        print('train time used %.1fmins' % ((end_time_tr - st_time_tr) / 60))
        print('test time used %.1fs' % (end_time_te - st_time_te))
        print('instance accuracy', elm.accuracy)
        print('mean class accuracy', elm.mean_cls_accuracy)





