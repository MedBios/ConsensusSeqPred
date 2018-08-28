import tflearn
from tflearn.layers.core import fully_connected as fc, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, global_avg_pool
from tflearn.layers.recurrent import lstm
from tflearn.layers.normalization import batch_normalization as bn
from tflearn.layers.embedding_ops import embedding
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
    required=True,
    help='What is the string length you want the network to take as input?')
parser.add_argument(
    '--train',
    type=str,
    default='y',
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
    '--num_epochs',
    type=int,
    default=15,
    help='How many epochs to train for. Default 15.')
parser.add_argument(
    '--learning_rate',
    type=float,
    default=.001,
    help='Learning rate for training. Default .001.')
parser.add_argument(
    '--view',
    type=bool,
    default=False,
    help='visualize all proteins together to see trends. Default False.')
parser.add_argument(
    '--embedding',
    type=int,
    default=1,
    help='embedding size. Default 1 (no embedding).')
parser.add_argument(
    '--numProteins',
    type=int,
    help='The number of proteins you want from UniProt. Default is all of them.')
parser.add_argument(
    '--view_letter_histogram',
    type=bool,
    default=False,
    help='View a histogram of all amino acids in proteins with keyword. Default False.')

args = parser.parse_args()
str_len = args.string_length
val_f = args.validation_percent
Train = args.train
num_classes = 26
keyword = args.keyword
num_epochs = args.num_epochs
lr = args.learning_rate
view = args.view
emb = args.embedding
numProteins = args.numProteins
name = 'string_length_' + str(str_len) + '_' + keyword + '_embedding_' + str(emb)
lhist = args.view_letter_histogram


if __name__ == '__main__':
    in_layer = input_data([None, 1, str_len*2+2, 1])
    indices = in_layer[:, 0, :2, 0]

    if emb > 1:
        lstm1 = lstm(embedding(in_layer[:, 0, 2:, 0], 26, emb),
                               300, return_seq=True)
    else:
        lstm1 = lstm(in_layer[:, 0, 2:, :], 300, return_seq=True)

    # lstm branch
    lstm2 = lstm(lstm1, 300, return_seq=True)
    lstm3 = lstm(lstm2, 300, return_seq=True)
    lstm4 = lstm(lstm3, 300)

    # cnn branch
    in_layer = bn(in_layer)
    conv1 = conv_2d(in_layer, 64, [1, 10], 1)
    norm1 = relu(bn(conv1))
    conv2 = conv_2d(norm1, 128, [1, 6], 2)
    norm2 = relu(bn(conv2))
    conv3 = conv_2d(norm2, 256, [1, 3], 2)
    norm3 = relu(bn(conv3))
    gap = tf.reshape(global_avg_pool(norm3), [-1, 256])

    # fully-connected branch
    fc_ind = fc(indices, 50, activation='tanh')
    fc_ind2 = fc(fc_ind, 50, activation='tanh')

    # merge lstm, conv, and fc layers
    merged = tf.concat([lstm4, gap, fc_ind2], 1)

    fc = fc(merged, num_classes, activation='softmax')
    net = regression(fc, optimizer='adam', loss='categorical_crossentropy',
                                                    learning_rate=lr)
    model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir='.')

    if Train in ['Y', 'y']:
        X = load_data(keyword, str_len, numProteins, view, lhist)
        X, Y, testX, testY = make_labels(X, val_f, num_classes)
        print(X.shape)
        X, testX = X[:, None, :, None], testX[:, None, :, None]
        h5save(X, Y, testX, testY, str_len, name + '_data.h5')

        os.system('tensorboard --logdir=. &')
        model.fit(X, Y, validation_set=(testX, testY), n_epoch=num_epochs,
                  shuffle=True, batch_size=256, show_metric=True,
                  snapshot_step=100,
              run_id=name)
        model.save(name)
    else:
        _, _, testX, testY = h5load(str_len, name + '_data.h5')

        if emb > 1:
            embDistance(name, emb)

        model.load(name)
        tflearn.config.init_training_mode()
        vizIndexWeights(name)
        cm = make_conf_mat(testX, testY, model, str_len, num_classes)
        cm2excel(cm, str_len, name)
        np.save('confusion_matrix_lstmcnn_' + name, cm)
