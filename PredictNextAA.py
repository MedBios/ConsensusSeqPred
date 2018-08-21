import tflearn
from tflearn.layers.core import fully_connected as fc, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, global_avg_pool
from tflearn.layers.recurrent import lstm
from tflearn.layers.normalization import batch_normalization as bn
from tflearn.activations import relu
import numpy as np
import os, sys
from data_utilities import *
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument(
    '--string_length',
    type=int,
    default=50,
    help='What is the string length you want the network to take as input? Default 50.')
parser.add_argument(
    '--train',
    type=str,
    required=True,
    help='y to train network (Default), n to load and test a trained network.')
parser.add_argument(
    '--validation_percent',
    type=float,
    default=0.25,
    help='What is the percent of samples to use for validation? Default 0.25.')
parser.add_argument(
    '--keyword',
    type=str,
    default='DNA-binding+antibody',
    help='What keyword do you want to query uniprot with? Default DNA-binding+antibody')
parser.add_argument(
    '--augmentation',
    type=bool,
    default=False,
    help='True for data augmentation, False for no augmentation. Default False')
parser.add_argument(
    '--num_epochs',
    type=int,
    default=15,
    help='How many epochs to train for. Default 15.')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=.001,
    help='Learning rate for training. Default .001.')

args = parser.parse_args()
str_len = args.string_length
val_f = args.validation_percent
Train = args.train
Augmentation = args.augmentation
num_classes = 25
keyword = args.keyword
num_epochs = args.num_epochs
lr = args.learning_rate


if __name__ == '__main__':
    in_layer = input_data([None, 1, str_len, 1])

    # lstm branch
    lstm1 = lstm(in_layer[:, 0, ...], 300, return_seq=True)
    lstm2 = lstm(lstm1, 300, return_seq=True)
    lstm3 = lstm(lstm2, 300, return_seq=True)
    lstm4 = lstm(lstm3, 300)

    # cnn branch
    conv1 = conv_2d(in_layer, 32, [1, 3], 1)
    norm1 = relu(bn(conv1))
    conv2 = conv_2d(norm1, 64, [1, 6], 2)
    norm2 = relu(bn(conv2))
    conv3 = conv_2d(norm2, 128, [1, 10], 2)
    norm3 = relu(bn(conv3))
    gap = tf.reshape(global_avg_pool(norm3), [-1, 128])

    # merge lstm and conv layers
    merged = tf.concat([lstm4, gap], 1)

    fc = fc(merged, num_classes, activation='softmax')
    net = regression(fc, optimizer='adam', loss='categorical_crossentropy',
                                                    learning_rate=lr)
    model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir='.')

    if Train in ['Y', 'y']:
        X = load_data(keyword, str_len, 20000)
        X, Y, testX, testY = make_labels(X, val_f, num_classes)
        X, testX = normalize(X, testX)
        X, testX = X[:, None, ...], testX[:, None, ...]
        h5save(X, Y, testX, testY, str_len, 'protein_predict_lstmcnn.h5')

        if Augmentation is True:
            X = augment(X)

        os.system('tensorboard --logdir=. &')
        model.fit(X, Y, validation_set=(testX, testY), n_epoch=num_epochs,
                  shuffle=True, batch_size=256, show_metric=True,
                  snapshot_step=100,
              run_id='Protein_predict_lstmcnn' + str(str_len))
        model.save('Protein_predict_lstmcnn' + str(str_len))
    else:
        _, _, testX, testY = h5load(str_len, 'protein_predict_lstmcnn.h5')
        model.load('Protein_predict_lstmcnn' + str(str_len))
        tflearn.config.init_training_mode()
        cm = make_conf_mat(testX, testY, model, str_len, num_classes)
        cm2excel(cm, str_len)
        np.save('confusion_matrix_lstmcnn' + str(str_len), cm)
