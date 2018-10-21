import tflearn
import numpy as np
import os, sys
from data_utilities_v2 import *
from model import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    'string_length',
    type=int,
    help='What is the string length you want the network to take as input?')
parser.add_argument(
    '--mode',
    type=str,
    default='train',
    choices=['train', 'test'],
    help="'train' to train a network, 'test' to test a trained one and do analysis.")
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
parser.add_argument(
    '--view_outputs',
    type=bool,
    default=False,
    help='Plot the outputs of the network as an image. Default False.')
parser.add_argument(
    '--letters_around',
    type=bool,
    default=False,
    help='Get the letters before and after every other letter.')
parser.add_argument(
    '--whiten',
    type=bool,
    default=False,
    help='True to whiten strings or False (default) to not whiten.')

args = parser.parse_args()
str_len = args.string_length
keyword = args.keyword
emb = args.embedding
name = 'string_length_' + str(str_len) + '_' + keyword + '_embedding_' + str(emb)
num_classes = 26


if __name__ == '__main__':
    # instantiate network
    protein_net = ProteinNet(str_len, emb, args.learning_rate, num_classes)
    model = protein_net.network()  # build model

    # load in and process data
    data_loader = process_data(keyword,
                               args.numProteins,
                               args.view_letter_histogram,
                               args.letters_around,
                               args.view,
                               args.whiten,
                               str_len,
                               args.validation_percent,
                               num_classes)

    # if train argument is true, load data, make labels, process data, and train
    if args.mode in ['train', 'Train']:
        X, Y, testX, testY = data_loader.load_data() # load and process data

        # Add extra dimensions for network
        X, testX = X[:, None, :, None], testX[:, None, :, None]

        # save dataset with this string length
        h5save(X, Y, testX, testY, str_len, name + '_data.h5')

        os.system('tensorboard --logdir=. &') # start tensorboard

        # perform training
        model.fit(X, # training sequences
                  Y,  # training labels
                  validation_set=(testX, testY),  # validation seqs/labels
                  n_epoch=args.num_epochs, # number of epochs
                  shuffle=True,  # shuffle data examples before each epoch
                  batch_size=256,  # batch size to use
                  show_metric=True,  # show val. acc/loss
                  snapshot_step=100,  # how often to validate
                  run_id=name)  # tensorboard name

        model.save(name) # save trained model

    else:  # if train is not true, load in a saved dataset and use it to do analysis
        _, _, testX, testY = h5load(str_len, name + '_data.h5')

        # if embedding used, calculate distance between each letter's embedding
        if emb > 1:
            print('Calculating Embedding Distances...')
            embDistance(name, emb, str_len)

        model.load(name) # load trained model
        tflearn.config.init_training_mode()

        # if view output map is True
        if args.view_outputs:
            X = get_uniprot_data(keyword)
            output_list2img(X, model, str_len)

        vizIndexWeights(name)  # view the index weights/biases

        # make a confusion matrix containing pairs of each letter
        cm = make_conf_mat(testX, testY, model, str_len, num_classes)
        cm2excel(cm, str_len, name)
        np.save('confusion_matrix_lstmcnn_' + name, cm)
