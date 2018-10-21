import tflearn
from tflearn.layers.core import fully_connected as fc, input_data
from tflearn.layers.estimator import regression
from tflearn.layers.conv import conv_2d, global_avg_pool, max_pool_2d
from tflearn.layers.recurrent import lstm
from tflearn.layers.normalization import batch_normalization as bn
from tflearn.layers.embedding_ops import embedding
from tflearn.activations import relu
import tensorflow as tf


class ProteinNet:
    def __init__(self, str_len, embedding_dim, learning_rate, num_classes):
        self.str_len = str_len
        self.emb = embedding_dim
        self.lr = learning_rate
        self.num_classes = num_classes


    def residual_block(self, input_, n_filters, filter_size, n_blocks, stride=1):
        strides = stride

        for block in range(n_blocks):
            n, h, w, c = input_.get_shape().as_list()
            if block > 0:
                strides = 1

            conv1 = conv_2d(input_, n_filters, filter_size, strides=strides, activation='linear')
            act1 = relu(bn(conv1))

            conv2 = conv_2d(act1, n_filters, filter_size, strides=1, activation='linear')
            act2 = bn(conv2)

            if stride != 1 and block == 0:
                input_ = max_pool_2d(input_, [1, 3], [1, stride])
            if n_filters != c:
                input_ = conv_2d(input_, n_filters, 1, activation='linear')

            input_ = relu(input_ + act2)

        return input_


    def network(self):
        in_layer = input_data([None, 1, self.str_len*2+2, 1])
        indices = in_layer[:, 0, :2, 0]

        if self.emb > 1:
            lstm1 = lstm(embedding(in_layer[:, 0, 2:, 0], 26, self.emb),
                                   300, return_seq=True)
        else:
            lstm1 = lstm(in_layer[:, 0, 2:, :], 300, return_seq=True)

        # lstm branch
        lstm2 = lstm(lstm1, 300, return_seq=True)
        lstm3 = lstm(lstm2, 300, return_seq=True)
        lstm4 = lstm(lstm3, 300)

        # cnn branch
        in_layer = bn(in_layer)
        conv1 = conv_2d(in_layer, 64, [1, 7], 1)
        norm1 = relu(bn(conv1))
        block1 = self.residual_block(norm1, 128, [1, 3], 2, stride=2)
        block2 = self.residual_block(block1, 256, [1, 3], 2, stride=2)
        block3 = self.residual_block(block2, 512, [1, 3], 2)
        block4 = self.residual_block(block3, 1024, [1, 3], 2)
        n_out_filters = block4.get_shape().as_list()[-1]
        gap = tf.reshape(global_avg_pool(block4), [-1, n_out_filters])

        # fully-connected branch
        fc_ind = fc(indices, 100, activation='tanh')
        fc_ind2 = fc(fc_ind, 100, activation='tanh')

        # merge lstm, conv, and fc layers
        merged = tf.concat([lstm4, gap, fc_ind2], 1)

        out = fc(merged, self.num_classes, activation='softmax') # output layer

        # describe optimization
        net = regression(out, optimizer='adam', loss='categorical_crossentropy',
                                                        learning_rate=self.lr)

        # build model
        model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir='.')

        return model
