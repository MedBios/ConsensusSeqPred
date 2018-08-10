import tflearn
from tflearn.layers.core import fully_connected as fc, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm
from tflearn.layers.normalization import batch_normalization as bn
import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
import numpy as np
import os, sys
from skimage.util import view_as_windows as vaw
from progressbar import ProgressBar
import h5py
import glob
from random import random
import xlwt
from xlwt import Workbook

str_len = 60
Train = False
Augmentation = True


def get_uniprot_data(kw, numxs=None):
    '''Goes to the uniprot website and searches for
       data with the keyword given. Returns the data
       found up to limit elements.'''

    url1 = 'http://www.uniprot.org/uniprot/?query='

    if numxs is None:
        url2 = '&columns=sequence&format=tab'
    else:
        url2 = '&columns=sequence&format=tab&limit='+str(numxs)

    query_complete = url1 + kw + url2
    request = Request(query_complete)
    response = urlopen(request)
    data = response.read()
    data = str(data, 'utf-8')
    data = data.split('\n')
    data = data[1:-1]
    data = list(map(lambda x:x.lower(), data))
    return data


def cut_and_label(seqs, length, padlen=None, save=False):
    if padlen is None:
        padlen = int(length * 0.8)

    x = np.zeros([0, length+1, 1])
    count = 0
    avg_len = []
    bar = ProgressBar()

    for seq in bar(seqs):
        seq_nums = []
        for letter in seq:
            seq_nums.append(max(ord(letter)-97, 0))
        avg_len.append(len(seq_nums))
        padded_seq = np.pad(np.asarray(seq_nums), (padlen, 0),
                            'constant', constant_values=22.)

        if padded_seq.size > length + 1:
            cut_seq = vaw(padded_seq, (length + 1, ))
            x = np.concatenate((x, cut_seq[..., None]))
            count += 1
        else:
            continue

    if save is True:
        f = h5py.File('Protein_prediction_data_' + str(length) + '_.h5', 'a')
        f.create_dataset('X', data=x)
        f.close()

    print('Used {} proteins. Avg. seq length was {}'.format(count, np.mean(avg_len)))
    plt.hist(avg_len, bins=50)
    plt.show()

    return x


def make_labels(X):
    Y = np.zeros([X.shape[0], int(np.amax(X)) + 1])
    for i in range(X.shape[0]):
        Y[i, int(X[i, -1, 0])] = 1.

    return X[:, :-1, :], Y


def augment(strings):
    for str in range(strings.shape[0]):
        if random() > 0.6:
            strings[str, ...] = strings[str, ::-1, :]
    return strings




if len(glob.glob('*' + str(str_len) + '_.h5')) == 0:
    X = get_uniprot_data('DNA-binding+antibody')
    X = cut_and_label(X, str_len, save=True)
else:
    h5file = h5py.File('Protein_prediction_data_' + str(str_len) + '_.h5', 'r')
    X = h5file['X']

X, Y = make_labels(X)
X = augment(X)


in_layer = input_data([None, str_len, 1])
lstm1 = lstm(in_layer, 300, return_seq=True)
lstm2 = lstm(lstm1, 300, return_seq=True)
lstm3 = lstm(lstm2, 300, return_seq=True)
lstm4 = lstm(lstm3, 300)
fc = fc(lstm4, int(np.amax(X)) + 1, activation='softmax')
net = regression(fc, optimizer='adam', loss='categorical_crossentropy',
                                                    learning_rate=0.0005)
model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir='.')


if Train is True:
    os.system('tensorboard --logdir=. &')
    model.fit(X, Y, validation_set=0.25, n_epoch=20, shuffle=True, batch_size=256,
          show_metric=True, snapshot_step=100,
          run_id='Protein_predict_' + str(str_len))
    model.save('Protein_predict_' + str(str_len))
else:
    model.load('Protein_predict_' + str(str_len))
    tflearn.config.init_training_mode()
    cm = np.zeros([25, 25])

    bar2 = ProgressBar()
    for i in bar2(range(X.shape[0])):
        x, y = X[i, ...], np.argmax(Y[i, :])
        pred = np.argmax(model.predict(x[None, ...])[0, ...])
        cm[y, pred] += 1

    np.save('confusion_matrix_' + str(str_len), cm)
