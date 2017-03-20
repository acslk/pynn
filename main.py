import numpy as np
import struct
import random
from array import array
from PIL import Image

import abc

TRAIN_DATA = 'train-images.idx3-ubyte'
TRAIN_LABEL = 'train-labels.idx1-ubyte'
TEST_DATA = 't10k-images.idx3-ubyte'
TEST_LABEL = 't10k-labels.idx1-ubyte'


def load_train_data():
    with open(TRAIN_LABEL, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        labels = np.array(array("B", file.read()))
    with open(TRAIN_DATA, 'rb') as file:
        magic, size, row, col = struct.unpack(">IIII", file.read(16))
        images = np.array(array("B", file.read()))
        images = images.reshape(size, row*col)
    return images, labels


class Model(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, data, labels):
        """Method documentation"""
        return

class MultiLayerModel(Model):

    def __init__(self, layers, nodes, batch_size, learn_rate):
        assert layers+1 == len(nodes)
        self.layers = layers
        self.nodes = nodes
        self.batch_size = batch_size
        self.lr = learn_rate
        self.weights = []
        for l in range(layers):
            input_dim = nodes[l]
            output_dim = nodes[l+1]
            self.weights.append(np.array([(random.random()-0.5)/100 for _ in range((input_dim+1)*output_dim)]).
                                reshape((input_dim + 1, output_dim)))
        self.out_exp = None  # has shape batch_no x output_dim, stores exponents for final softmax
        self.out_sum = None  # has shape batch_no, stores sum of exponents of out_exp for each item
        self.last_batch = []    # has input values for each layer
        self.last_label = None  # has shape batch_no

    # forward pass, return the loss
    #@profile
    def forward(self, batch_input, label):
        data_size = batch_input.shape[0]
        self.last_batch = []
        self.last_label = label
        batch_input = np.hstack((batch_input, np.ones((data_size, 1))))
        self.last_batch.append(batch_input)

        # go through intermediate layers
        for layer in range(self.layers-1):
            batch_input = np.dot(batch_input, self.weights[layer])
            # leaky RElu activation
            batch_input[batch_input < 0] *= 0.05
            batch_input = np.hstack((batch_input, np.ones((data_size, 1))))
            self.last_batch.append(batch_input)

        # last layer
        out = np.dot(batch_input, self.weights[self.layers-1])
        self.out_exp = 2 ** out
        self.out_sum = np.sum(self.out_exp, axis=1)
        loss = np.average(1 - self.out_exp[([i for i in range(data_size)], label)] / self.out_sum)
        accuracy = np.average(np.equal(np.argmax(self.out_exp, axis=1), label))
        return loss, accuracy

    #@profile
    def back_prop(self):
        old_weights = []
        for weight in self.weights:
            old_weights.append(weight.copy())
        # out_exp has shape (batch_size, output_dim)
        for ino, out_e in enumerate(self.out_exp):
            row_sum = self.out_sum[ino]
            lable_no = self.last_label[ino]
            loss_dev = np.zeros(self.nodes[self.layers])
            for i, e in enumerate(out_e):
                if i == lable_no:
                    loss_dev[i] = (e*row_sum - e**2)/row_sum**2
                else:
                    loss_dev[i] = -e*out_e[lable_no]/row_sum**2
            for layer in reversed(range(self.layers)):
                w_diff = loss_dev * self.last_batch[layer][ino].reshape((-1, 1))*self.lr
                self.weights[layer] += w_diff
                loss_dev = np.dot(loss_dev, old_weights[layer].T)[:-1]
                #loss_dev[self.last_batch[layer][ino][:-1] < 0] *= 0.05

    #@profile
    def train(self, data, labels):
        last_loss = 1
        # number of times to go through entire train dataset
        for l in range(100):
            for i in range(50000//self.batch_size):
                batch_data = data[i*self.batch_size:(i+1)*self.batch_size]
                batch_label = labels[i*self.batch_size:(i+1)*self.batch_size]
                loss, accuracy = self.forward(batch_data, batch_label)
                self.back_prop()
                #print('iter: {}, loss: {}, accuracy: {}'.format(i, loss, accuracy))
            loss, accuracy = self.forward(data[:50000], labels[:50000])
            print('TRAIN')
            print('iter: {}, loss: {}, accuracy: {}'.format(l, loss, accuracy))
            loss, accuracy = self.forward(data[50000:], labels[50000:])
            print('DEV')
            print('iter: {}, loss: {}, accuracy: {}'.format(l, loss, accuracy))
            if loss < last_loss/2:
                self.lr /= 10
                last_loss = loss


if __name__ == '__main__':
    images, labels = load_train_data()
    images = images/255
    dev_start = 50000
    model = MultiLayerModel(2, [28*28, 128, 10], 10, 0.01)
    model.train(images, labels)
    # model.eval(images[dev_start:])
    # image_out = Image.fromarray(images[0].reshape((28,28)))
    # image_out.save('a.png')
