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

    @abc.abstractclassmethod
    def eval(self, data):
        return


class SingleLayerModel(Model):

    def __init__(self, input_dim, batch_size, output_dim):
        self.input_d = input_dim
        self.output_d = output_dim
        self.batch_size = batch_size
        self.weight = np.array([(random.random()-0.5)/100 for _ in range((input_dim+1)*output_dim)]).reshape((input_dim + 1, output_dim))
        self.out_exp = np.zeros(10)
        self.out_sum = 0
        self.range_array = [i for i in range(self.batch_size)]

    # forward pass, return the loss
    def forward(self, batch_input, label):
        print(batch_input.shape)
        batch_input = np.hstack((batch_input, np.ones((self.batch_size, 1))))
        out = np.dot(batch_input, self.weight)
        self.out_exp = 2 ** out
        self.out_sum = np.sum(self.out_exp, axis=1)
        return 1 - self.out_exp[(self.range_array, label)] / self.out_sum


    def back_prop(self):

        return self.weight

    def train(self, data, labels):
        loss = self.forward(data[0:10], labels[0:10])
        print(loss)

    def eval(self, data):
        self.forward(data[0:self.batch_size], np.zeros(self.batch_size, dtype=np.int))
        print(np.argmax(self.out_exp, axis=1))

if __name__ == '__main__':
    images, labels = load_train_data()
    dev_start = 50000
    model = SingleLayerModel(28*28, 10, 10)
    model.train(images[:dev_start], labels)
    model.eval(images[dev_start:])
    image_out = Image.fromarray(images[0].reshape((28,28)))
    image_out.save('a.png')
