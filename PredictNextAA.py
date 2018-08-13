import tflearn
from tflearn.layers.core import fully_connected as fc, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm
from tflearn.layers.normalization import batch_normalization as bn
import numpy as np
import os, sys
from data_utilities import *
import argparse

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

args = parser.parse_args()
str_len = args.string_length
val_f = args.validation_percent
Train = args.train
Augmentation = args.augmentation
num_classes = 25
keyword = args.keyword
num_epochs = args.num_epochs



if __name__ == '__main__':
    in_layer = input_data([None, str_len, 1])
    lstm1 = lstm(in_layer, 300, return_seq=True)
    lstm2 = lstm(lstm1, 300, return_seq=True)
    lstm3 = lstm(lstm2, 300, return_seq=True)
    lstm4 = lstm(lstm3, 300)
    fc = fc(lstm4, num_classes, activation='softmax')
    net = regression(fc, optimizer='adam', loss='categorical_crossentropy',
                                                        learning_rate=0.0005)
    model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir='.')

    if Train in ['Y', 'y']:
        X = load_data(keyword, str_len)
        X, Y, testX, testY = make_labels(X, val_f, num_classes)
        X, testX = normalize(X, testX)
        h5save(X, Y, testX, testY, str_len)

        if Augmentation is True:
            X = augment(X)
        os.system('tensorboard --logdir=. &')
        model.fit(X, Y, validation_set=(testX, testY), n_epoch=num_epochs,
                  shuffle=True, batch_size=256, show_metric=True,
                  snapshot_step=100,
              run_id='Protein_predict_' + str(str_len))
        model.save('Protein_predict_' + str(str_len))
    else:
        X, _, testX, testY = h5load(str_len)
        _, testX = normalize(X, testX)
        model.load('Protein_predict_' + str(str_len))
        tflearn.config.init_training_mode()
        cm = make_conf_mat(testX, testY, model, str_len, num_classes)
        cm2excel(cm, str_len)
        np.save('confusion_matrix_' + str(str_len), cm)
