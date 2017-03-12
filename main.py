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


class SingleLayerModel(Model):

    def __init__(self, input_dim, batch_size, output_dim, learn_rate):
        self.input_d = input_dim
        self.output_d = output_dim
        self.batch_size = batch_size
        self.lr = learn_rate
        self.weight = np.array([(random.random()-0.5)/100 for _ in range((input_dim+1)*output_dim)]).reshape((input_dim + 1, output_dim))
        self.out_exp = np.zeros(10)
        self.out_sum = 0
        self.range_array = [i for i in range(self.batch_size)]
        self.last_batch = None
        self.last_label = None

    # forward pass, return the loss
    def forward(self, batch_input, label):
        batch_input = np.hstack((batch_input, np.ones((self.batch_size, 1))))
        self.last_batch = batch_input
        self.last_label = label
        out = np.dot(batch_input, self.weight)
        self.out_exp = 2 ** out
        self.out_sum = np.sum(self.out_exp, axis=1)
        return 1 - self.out_exp[(self.range_array, label)] / self.out_sum

    def back_prop(self):
        # out_exp has shape (batch_size, output_dim)
        for ino, out_e in enumerate(self.out_exp):
            row_sum = self.out_sum[ino]
            lable_no = self.last_label[ino]
            loss_dev = np.zeros(self.output_d)
            for i, e in enumerate(out_e):
                if i == lable_no:
                    loss_dev[i] = (e*row_sum - e**2)/row_sum**2
                else:
                    loss_dev[i] = -e*out_e[lable_no]/row_sum**2
                self.weight += loss_dev * self.last_batch[ino].reshape((-1, 1))*self.lr

    def eval_label(self, data, data_label):
        data_size = data.shape[0]
        batch_input = np.hstack((data, np.ones((data_size, 1))))
        out = np.dot(batch_input, self.weight)
        out_exp = 2 ** out
        out_sum = np.sum(out_exp, axis=1)
        accuracy = np.equal(np.argmax(out_exp, axis=1), data_label).sum() / data_size
        loss = (1 - out_exp[([i for i in range(data_size)], data_label)] / out_sum).sum() / data_size
        return loss, accuracy

    def train(self, data, labels):
        last_loss = 1
        for l in range(100):
            for i in range(5000):
                batch_data = data[i*self.batch_size:(i+1)*self.batch_size]
                batch_label = labels[i*self.batch_size:(i+1)*self.batch_size]
                loss = self.forward(batch_data, batch_label)
                self.back_prop()
                loss = np.average(loss)
                accuracy = np.equal(np.argmax(self.out_exp, axis=1), batch_label).sum()/10
                #print('iter: {}, loss: {}, accuracy: {}'.format(i, loss, accuracy))
            loss, accuracy = self.eval_label(data[:50000], labels[:50000])
            print('TRAIN')
            print('iter: {}, loss: {}, accuracy: {}'.format(l, loss, accuracy))
            loss, accuracy = self.eval_label(data[50000:], labels[50000:])
            print('DEV')
            print('iter: {}, loss: {}, accuracy: {}'.format(l, loss, accuracy))
            if loss < last_loss/2:
                self.lr /= 10
                last_loss = loss


class MultiLayerModel(Model):

    def __init__(self, layers, nodes, batch_size, learn_rate):
        self.input_d = input_dim
        self.output_d = output_dim
        self.batch_size = batch_size
        self.lr = learn_rate
        self.weight = np.array([(random.random()-0.5)/100 for _ in range((input_dim+1)*output_dim)]).reshape((input_dim + 1, output_dim))
        self.out_exp = np.zeros(10)
        self.out_sum = 0
        self.range_array = [i for i in range(self.batch_size)]
        self.last_batch = None
        self.last_label = None

    # forward pass, return the loss
    def forward(self, batch_input, label):
        batch_input = np.hstack((batch_input, np.ones((self.batch_size, 1))))
        self.last_batch = batch_input
        self.last_label = label
        out = np.dot(batch_input, self.weight)
        self.out_exp = 2 ** out
        self.out_sum = np.sum(self.out_exp, axis=1)
        return 1 - self.out_exp[(self.range_array, label)] / self.out_sum

    def back_prop(self):
        # out_exp has shape (batch_size, output_dim)
        for ino, out_e in enumerate(self.out_exp):
            row_sum = self.out_sum[ino]
            lable_no = self.last_label[ino]
            loss_dev = np.zeros(self.output_d)
            for i, e in enumerate(out_e):
                if i == lable_no:
                    loss_dev[i] = (e*row_sum - e**2)/row_sum**2
                else:
                    loss_dev[i] = -e*out_e[lable_no]/row_sum**2
                self.weight += loss_dev * self.last_batch[ino].reshape((-1, 1))*self.lr

    def eval_label(self, data, data_label):
        data_size = data.shape[0]
        batch_input = np.hstack((data, np.ones((data_size, 1))))
        out = np.dot(batch_input, self.weight)
        out_exp = 2 ** out
        out_sum = np.sum(out_exp, axis=1)
        accuracy = np.equal(np.argmax(out_exp, axis=1), data_label).sum() / data_size
        loss = (1 - out_exp[([i for i in range(data_size)], data_label)] / out_sum).sum() / data_size
        return loss, accuracy

    def train(self, data, labels):
        last_loss = 1
        for l in range(100):
            for i in range(5000):
                batch_data = data[i*self.batch_size:(i+1)*self.batch_size]
                batch_label = labels[i*self.batch_size:(i+1)*self.batch_size]
                loss = self.forward(batch_data, batch_label)
                self.back_prop()
                loss = np.average(loss)
                accuracy = np.equal(np.argmax(self.out_exp, axis=1), batch_label).sum()/10
                #print('iter: {}, loss: {}, accuracy: {}'.format(i, loss, accuracy))
            loss, accuracy = self.eval_label(data[:50000], labels[:50000])
            print('TRAIN')
            print('iter: {}, loss: {}, accuracy: {}'.format(l, loss, accuracy))
            loss, accuracy = self.eval_label(data[50000:], labels[50000:])
            print('DEV')
            print('iter: {}, loss: {}, accuracy: {}'.format(l, loss, accuracy))
            if loss < last_loss/2:
                self.lr /= 10
                last_loss = loss


if __name__ == '__main__':
    images, labels = load_train_data()
    images = images/255
    dev_start = 50000
    model = SingleLayerModel(28*28, 10, 10, 0.01)
    model.train(images, labels)
    # model.eval(images[dev_start:])
    # image_out = Image.fromarray(images[0].reshape((28,28)))
    # image_out.save('a.png')
