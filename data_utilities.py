import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
import numpy as np
import os, sys
import xlwt
from xlwt import Workbook
from skimage.util import view_as_windows as vaw
from progressbar import ProgressBar
import glob
from random import random
import h5py
from scipy.misc import imshow
from tensorflow.python import pywrap_tensorflow
from math import factorial


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


def letter_frequency(data):
    l = np.zeros([26, ])
    wb = Workbook()
    sheet1 = wb.add_sheet('letters')
    for d in data:
        for D in d:
            l[max(ord(D)-97, 0)] += 1

    for i in range(l.size):
        sheet1.write(i, 0, chr(i + 97))
        sheet1.write(i, 1, l[i])

    print(l)
    wb.save('LetterFrequency.xlsx')

    return


def letter2num(seqs):
    seqList = []

    for seq in seqs:
        seqNums = []
        for letter in seq:
            seqNums.append(max(ord(letter)-97, 0))
        seqList.append(seqNums)

    return seqList


def cutStrings(seqs, length):
    X = np.zeros([0, length*2+3])
    count = 0
    bar = ProgressBar()

    for seq in bar(seqs):
        paddedSeq = np.pad(np.asarray(seq), (length, length),
                           'constant', constant_values=23.)

        if paddedSeq.size > length*2:
            cutSeq = vaw(paddedSeq, (length*2+1, ))
            label = np.ones([cutSeq.shape[0], ]) * cutSeq[:, length]
            cutSeq = np.delete(cutSeq, length, axis=1)
            indLabel = np.ones([cutSeq.shape[0], 1]) * np.arange(cutSeq.shape[0])[:, None]
            cutSeq = np.concatenate((indLabel, indLabel[::-1, :], cutSeq, label[:, None]), 1)
            X = np.concatenate((X, cutSeq), 0)
            count += 1
        else:
            continue

    print('Used {} proteins.'.format(count))
    return X


def make_labels(X, validation_size, num_classes):
    Y = np.zeros([X.shape[0], num_classes])
    for i in range(X.shape[0]):
        Y[i, int(X[i, -1])] = 1.

    n_val = int(X.shape[0]*validation_size)
    rtest = np.random.randint(0, X.shape[0], n_val)
    testX, X = X[rtest, ...], np.delete(X, rtest, axis=0)
    testY, Y = Y[rtest, :], np.delete(Y, rtest, axis=0)

    return X[:, :-1], Y, testX[:, :-1], testY


def lenSort(seqs, ascending=True):
  for i in range(1, len(seqs)):
    current, prevInd = seqs[i], i - 1

    if ascending:
      while prevInd >= 0 and len(seqs[prevInd]) > len(current):
        seqs[prevInd+1] = seqs[prevInd]
        prevInd -= 1
    else:
      while prevInd >= 0 and len(seqs[prevInd]) < len(current):
        seqs[prevInd+1] = seqs[prevInd]
        prevInd -= 1

    seqs[prevInd+1] = current

  return seqs


def list2img(X):
    X = lenSort(X, ascending=False)
    maxLength = len(X[0])
    strings = np.zeros([len(X), maxLength])
    count = 0

    for x in X:
        padlen = maxLength - len(x)
        paddedString = np.pad(np.asarray(x), (0, padlen), 'constant', constant_values=23.)
        strings[count, :paddedString.size] = paddedString
        count += 1

    imshow(strings)
    return


def load_data(kw, str_len, numxs, view, letter_freq):
    X = get_uniprot_data(kw, numxs)
    if letter_freq:
        letter_frequency(X)
    
    X = letter2num(X)
    if view:
        list2img(X)

    X = cutStrings(X, str_len)

    return X


def make_conf_mat(X, Y, model, string_length, num_classes):
    cm = np.zeros([num_classes, num_classes])

    bar2 = ProgressBar()
    for i in bar2(range(X.shape[0])):
        x, y = X[i, ...], np.argmax(Y[i, :])
        pred = np.argmax(model.predict(x[None, ...])[0, ...])
        cm[y, pred] += 1

    fig = plt.figure()
    a1 = fig.add_subplot(111)
    a1.set_title('confusion_matrix_' + str(string_length))
    a1.set_xlabel('Predicted')
    a1.set_ylabel('Actual')
    a1.imshow(cm)
    plt.show()

    return cm

def embDistance(filename, dims):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    embeddingWeights = reader.get_tensor('Embedding/W')
    wb = Workbook()
    emb1 = wb.add_sheet('embedding_distances')
    count = 0
    noLetters = ['j', 'o', 'u', 'x', 'z', 'b']

    for i in range(embeddingWeights.shape[0]-1, -1, -1):
        for j in range(i-1, -1, -1):
            if chr(i+97) not in noLetters and chr(j+97) not in noLetters:
                w1, w2 = embeddingWeights[i, :], embeddingWeights[j, :]
                emb1.write(count, 0, float(np.mean((w1 - w2)**2)))
                emb1.write(count, 1, chr(i+97) + chr(j+97))
                count += 1

    wb.save('embedding_distances_' + filename + '.xlsx')
    return


def cm2excel(cm, string_length, name):
    wb = Workbook()
    sheet1 = wb.add_sheet('confusion_matrix')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            sheet1.write(j, i, cm[i, j])
    wb.save('Confusion_matrix_' + name + '.xlsx')
    return


def vizIndexWeights(filename):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    fc1_W = reader.get_tensor('FullyConnected/W')
    fc1_b = reader.get_tensor('FullyConnected/b')

    fig = plt.figure()
    a1 = fig.add_subplot(221)
    a2 = fig.add_subplot(222)
    a3 = fig.add_subplot(223)
    a4 = fig.add_subplot(224)

    a1.set_ylabel('Frequency')
    a1.set_xlabel('Weight/Bias Value')
    a1.set_title('Index Relative to First Position')

    a2.set_ylabel('Frequency')
    a2.set_xlabel('Weight/Bias Value')
    a2.set_title('Index Relative to Last Position')

    a3.set_xlabel('Index Relativity')
    a3.set_ylabel('Median Weight Value')

    a4.set_xlabel('Index Relativity')
    a4.set_ylabel('Median Absolute Weight Value')

    a1.hist(fc1_W[0, :], bins=10)
    a1.hist(fc1_b, bins=10, fc=(1, 0, 0, 0.4))

    a2.hist(fc1_W[1, :], bins=10)
    a2.hist(fc1_b, bins=10, fc=(1, 0, 0, 0.4))

    a3.bar(['First', 'Last'], np.median(fc1_W, 1), width=0.35)
    a4.bar(['First', 'Last'], np.median(np.absolute(fc1_W), 1), width=0.35)

    plt.tight_layout()
    plt.show()

    return



def h5save(X, Y, testX, testY, string_length, save_name):
    h5f = h5py.File(save_name, 'a')
    h5f.create_dataset('testX', data=testX)
    h5f.create_dataset('testY', data=testY)
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('Y', data=Y)
    h5f.close()
    return


def h5load(string_length, name):
    h5f = h5py.File(name, 'r')
    x, y = h5f['X'], h5f['Y']
    tx, ty = h5f['testX'], h5f['testY']
    return x, y, tx, ty
