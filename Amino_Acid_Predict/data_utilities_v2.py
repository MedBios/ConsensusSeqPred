import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
import numpy as np
import os, sys, xlwt, h5py, tflearn
from xlwt import Workbook
from skimage.util import view_as_windows as vaw
from progressbar import ProgressBar
import glob
from random import random
from scipy.misc import imshow
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.tensorboard.plugins import projector
from numpy.fft import fft, ifft
import tensorflow as tf



class process_data:
    def __init__(self, kw, numxs, letter_freq, letters_around, view, white,
                 string_length, val_num, num_classes):
        self.kw = kw
        self.numxs = numxs
        self.letter_freq = letter_freq
        self.letters_around = letters_around
        self.view = view
        self.white = white
        self.str_len = string_length
        self.validation_num = val_num
        self.num_classes = num_classes


    def get_uniprot_data(self):
        '''Goes to the uniprot website and searches for
           data with the keyword given. Returns the data
           found up to limit elements.'''

        url1 = 'http://www.uniprot.org/uniprot/?query='  # first part of URL

        # make 2nd part of URL depending if number of proteins is specified or not
        if self.numxs is None:
            url2 = '&columns=sequence&format=tab'
        else:
            url2 = '&columns=sequence&format=tab&limit='+str(numxs)

        # Query uniprot with keyword and fetch data
        query_complete = url1 + self.kw + url2
        request = Request(query_complete)
        response = urlopen(request)
        data = response.read()
        data = str(data, 'utf-8')
        data = data.split('\n')
        data = data[1:-1]
        data = list(map(lambda x:x.lower(), data))

        return data


    def letter_frequency(self, data):
        l = np.zeros([26, ])
        wb = Workbook()
        sheet1 = wb.add_sheet('letters')
        for d in data:
            for D in d:
                l[max(ord(D)-97, 0)] += 1

        for i in range(l.size):
            sheet1.write(i, 0, chr(i + 97))
            sheet1.write(i, 1, l[i])

        wb.save('LetterFrequency.xlsx')


    def get_letters_around(self, data):
        ''' Gets the amino acids directly before and after each letter.
            Args:
                 data: A list of lists composed of protein strings.
        '''

        store_vals = np.zeros([26, 26*2+1]) # create a matrix to store vals

        for seq in data:
            for idx, letter in enumerate(seq):
                if idx != 0:
                    store_vals[ord(letter)-97, ord(seq[idx-1])-97] += 1.
                if idx != len(seq)-1:
                    store_vals[ord(letter)-97, 27+(ord(seq[idx+1])-97)] += 1.

        wb = Workbook()  # create an excel workbook object
        sheet1 = wb.add_sheet('Letters Before and After')
        [sheet1.write(row, 26, chr(row+97)) for row in list(range(26))]

        for x in range(store_vals.shape[0]):
            for y in range(store_vals.shape[1]):
                if y != 26:
                    sheet1.write(x, y, store_vals[x, y])

        wb.save('Letters_Around_' + self.kw + '.xlsx')



    def letter2num(self, seqs):
        seqList = []

        for seq in seqs:
            seqNums = []
            for letter in seq:
                seqNums.append(max(ord(letter)-97, 0))
            seqList.append(seqNums)

        return seqList


    def lenSort(self, seqs, ascending=True):
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


    def whiten(self, string):
        str_fft = fft(string, axis=1)
        spectrum = np.sqrt(np.mean(np.absolute(str_fft) ** 2))
        return np.absolute(ifft(str_fft * (1. / spectrum), axis=1))


    def list2img(self, X):
        X = self.lenSort(X, ascending=False)
        maxLength = len(X[0])
        strings = np.zeros([len(X), maxLength])
        count = 0

        for x in X:
            padlen = maxLength - len(x)
            x = np.asarray(x)

            paddedString = np.pad(x, (0, padlen), 'constant', constant_values=23.)
            strings[count, :paddedString.size] = paddedString
            count += 1

        if self.white:
            strings -= np.mean(strings, 0)
            strings = self.whiten(strings)

        imshow(strings)


    def cutStrings(self, seqs):
        print('Preparing Data...')
        X = np.zeros([0, self.str_len*2+3])
        count = 0
        bar = ProgressBar()

        for seq in bar(seqs):
            seq = np.asarray(seq)
            paddedSeq = np.pad(seq, (self.str_len, self.str_len),
                               'constant', constant_values=23.)

            if paddedSeq.size > self.str_len*2:
                cutSeq = vaw(paddedSeq, (self.str_len*2+1, ))
                label = np.ones([cutSeq.shape[0], ]) * cutSeq[:, self.str_len]
                cutSeq = np.delete(cutSeq, self.str_len, axis=1)
                indLabel = np.ones([cutSeq.shape[0], 1]) * np.arange(cutSeq.shape[0])[:, None]
                cutSeq = np.concatenate((indLabel, indLabel[::-1, :], cutSeq, label[:, None]), 1)
                X = np.concatenate((X, cutSeq), 0)
                count += 1
            else:
                continue

        print('Used {} proteins.'.format(count))
        return X


    def make_labels(self, X):
        Y = np.zeros([X.shape[0], self.num_classes])
        for i in range(X.shape[0]):
            Y[i, int(X[i, -1])] = 1.

        n_val = int(X.shape[0]*self.validation_num)
        rtest = np.random.randint(0, X.shape[0], n_val)
        testX, X = X[rtest, ...], np.delete(X, rtest, axis=0)
        testY, Y = Y[rtest, :], np.delete(Y, rtest, axis=0)

        return X[:, :-1], Y, testX[:, :-1], testY


    def load_data(self):
        X = self.get_uniprot_data()
        if self.letter_freq:
            self.letter_frequency(X)
        if self.letters_around:
            self.get_letters_around(X)

        X = self.letter2num(X)
        if self.view:
            self.list2img(X)

        X = self.cutStrings(X)

        return self.make_labels(X)



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


def viz_embedding(tensor, string_length, emb_size):

    tb_dir = os.getcwd()
    sess2 = tf.Session()
    sess2.run(tensor.initializer)
    summary_writer = tf.summary.FileWriter(tb_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor.name
    embedding.metadata_path = os.path.join(tb_dir, 'metadata.tsv')
    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([tensor])
    name = str(string_length) + '_' + str(emb_size)
    saver.save(sess2, os.path.join(tb_dir, name + tensor.name + '.ckpt'), 1)

    with open(os.path.join(tb_dir, 'metadata.tsv'),'w') as f:
        f.write("Index\tLabel\n")
        for index in range(26):
            f.write("%d\t%s\n" % (index,chr(index+97)))
    f.close()


def embDistance(filename, dims, string_length):
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    embeddingWeights = reader.get_tensor('Embedding/W')
    emb_layer = tf.Variable(embeddingWeights, name='emb_' + str(dims) + '_' + str(string_length))
    viz_embedding(emb_layer, string_length, dims)

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


def output_list2img(X, model, string_length):
    X = letter2num(X)
    bar = ProgressBar()

    for i in bar(range(len(X))):
        x = X[i]
        x_array = np.asarray(x)
        x_array = np.pad(x_array, (string_length, string_length), 'constant',
                         constant_values=23.)

        for j in range(string_length, x_array.size-string_length):
            x_prev = x_array[j-string_length:j]
            x_post = x_array[j+1:j+1+string_length]
            inds = np.array([j, x_array.size-j])
            x_in = np.concatenate((inds, x_prev, x_post))
            y_hat = model.predict(x_in[None, None, :, None])
            x[j - string_length] = np.argmax(y_hat)

        X[i] = x

    list2img(X)
    return



    # seq_lens = []
    # for x in X:
    #     seq_lens.append(len(x))
    #
    # X = letter2num(X)
    # X = cutStrings(X, string_length)
    # masterlist = []
    # count = 0
    #
    # print('Making output image...')
    # bar = ProgressBar()
    #
    # sublist = []
    # for i in bar(range(X.shape[0])):
    #     if count > seq_lens[0] - 1:
    #         masterlist.append(sublist)
    #         del seq_lens[0]
    #         sublist = []
    #         count = 0
    #
    #     x = X[i, :-1]
    #     y_hat = np.argmax(model.predict(x[None, None, :, None])[0, :])
    #     sublist.append(y_hat)
    #     count += 1


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
