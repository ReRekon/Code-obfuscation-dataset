# -*- coding: UTF-8 -*-
import os
import sys
import time

from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim

from gnnmodels.mergeModel import MergeClassifier

from ParameterConfig import ParameterConfig
from consDataset import consDataset

from train_eval import model_train, model_evaluate, plot_train_validation_acc_loss, metric_predictions

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

device = 'cuda' if th.cuda.is_available() else 'cpu'
device = th.device(device)
ParameterConfig.device = device
print(device)


def run_whole_procedure(corpus_path, dic_file_path, label_maps=None, node_vec_stg=None, token_vec_stg=None):
    """
    execute the whole procedure
    :param corpus_path: path of the dataset directory
    :param dic_file_path: 字典路径
    :param label_maps:   label的map
    :param node_vec_stg: 训练结点的模型
    :return:
    """

    # step 1: load data
    print('loading cfg data...')
    db = consDataset(corpus_path, dic_file_path, label_maps)
    # split into train and test set
    db_size = len(db)
    train_size = int((1 - ParameterConfig.dataset_split_ratio) * db_size)
    test_size = db_size - train_size
    trainset, testset = th.utils.data.random_split(db, [train_size, test_size])
    print(
        'dataset size: {:d} CFGs, {:d} for training and {:d} for testing'.format(db_size, len(trainset), len(testset)))

    fig_prefix = 'result/' + '#' + node_vec_stg
    # step 2: construct and train the model
    print('setting up the model...')
    num_classes = len(label_maps)

    model = MergeClassifier(128, 128,
                            num_classes, db.embeddings_matrix, node_vec_stg=node_vec_stg, text_vec_stg=token_vec_stg,
                            device=device, layer_num=ParameterConfig.GCN_Layer_Num,
                            dp_rate=ParameterConfig.GCN_DP_RATE).to(device)

    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=ParameterConfig.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=ParameterConfig.lr_decay)
    # train the model
    print('training the model')
    start = time.time()
    model_stats, history = model_train(fig_prefix, trainset, model, loss_func, optimizer, scheduler, testset)
    plot_train_validation_acc_loss(fig_prefix, history)
    end = time.time()
    print('neural-net training takes: %s seconds' % (end - start))
    # model is got and stored
    # th.save(model, model_store_path)

    # step 3: evaluate the trained model on the test data
    print('evaluating on the test-set the trained model')
    # model = init_model.load_state_dict(th.load(model_stats)).to(device)
    model.load_state_dict(th.load(model_stats))
    model.eval()
    preds, ground_truth = model_evaluate(model, testset, loss_func, fig_prefix)
    metric_predictions(preds, ground_truth, fig_prefix)

    th.cuda.empty_cache()


if __name__ == '__main__':
    corpus_base_path = 'inputs/input/'
    dic_base_path = 'inputs/PhaseII-ins2vec/'

    # 1 表示工具识别, 其他的表示算法识别
    trainType = 1

    if trainType == 1:
        # construct the label map
        label_maps = {
            "c-code-obfusactor": 0,
            "cobf-1.06": 1,
            "cobfu": 2,
            "cpp-guard": 3,
            "cxx-obfus": 4,
            "obfuscator-in-Python": 5,
            "tigress": 6
        }
        dataset_name = 'confusion_small'
        dic_file_path = dic_base_path + 'w2v_model.dic'
    else:
        label_maps = {
            "encodeArithmetic1": 0,
            "encodeLiterals1": 1,
            "flatten1": 2,
            "virt1": 3
        }
        dataset_name = 'tigress_arithmetic3'
        dic_file_path = dic_base_path + 'w2v_model_tigress.dic'

    node_vec_stg = 'TextCNN'
    token_vec_stg = 'TextRCNN'

    hyper_log_prefix = 'result/'
    ParameterConfig.log_config(hyper_log_prefix)

    print('-------------------' + dataset_name + '-------------------')
    corpus_path = corpus_base_path + dataset_name

    run_whole_procedure(corpus_path, dic_file_path, label_maps, node_vec_stg, token_vec_stg)
