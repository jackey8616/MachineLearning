from os import listdir
from os.path import isfile, join
import sys
import tensorflow as tf
import pandas as pd

class ModelNFMD:

    def __init__(self, amount=10):
        print('Model initing...')
        self.modelFile = './model_saves/Tensorflow/TenLogisticReg.ckpt'
        self.columns = ['FileID', 'CustomerID', 'QueryTS', 'ProductID']
        self.csvFolder = './data/train_data/'
        self.csvFiles = [f for f in listdir(self.csvFolder) if isfile(join(self.csvFolder, f))]
        self.amount = amount
        
        self.fileID = tf.placeholder(tf.string, [None, 1])
        self.customerID = tf.placeholder(tf.string, [None, 1])
        self.queryTS = tf.placeholder(tf.float32, [None, 1])
        self.productID = tf.placeholder(tf.string, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        print('Variable defined.')
   
        self.concatFile()

    def train(self, epochs=100):
        print('Training...')
        train_fileID = self.train_data[:, 0:1]
        train_customerID = self.train_data[:, 1:2]
        train_queryTS = self.train_data[:, 2:3]
        train_productID = self.train_data[:, 3:4]
        print('Seperated column from train data.')
        train_X = self.train_data[:, :1]
        train_y = self.train_data[:, -1:]
        lenSymbol = len(train_X[0])
        lenData = len(train_X)
        print('Size of train_X: {}x{}'.format(lenData, lenSymbol))
        print('Size of train_y: {}x{}'.format(len(train_y), len(train_y[0])))


        W = tf.Variable(tf.random_normal([1, 1]))
        b = tf.Variable(tf.zeros([1, 1]) + 1)
        #z = tf.matmul(X, W) + b
        #o = tf.sigmoid(z)
        #cross_entropy = tf.reduce_mean(y * -tf.log(o) + (1 - y) * -tf.log(1 - o))
        cross_entropy = tf.reduce_mean(tf.square(W + b - self.y))
        train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        optimal_W = None
        optimal_b = None
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(epochs):
                sess.run(train, feed_dict={self.fileID: train_fileID, self.customerID: train_customerID, self.queryTS: train_queryTS, self.productID: train_productID, self.y:train_y})
                #if i % 50 == 0:
                print(i, sess.run(cross_entropy, feed_dict={self.fileID: train_fileID, self.customerID: train_customerID, self.queryTS: train_queryTS, self.productID: train_productID, self.y:train_y}))
                #print(self.train_data)
            optimal_W = sess.run(W)
            optimal_b = sess.run(b)
            self.saveModel(sess)

    def saveModel(self, sess):
        tf.train.Saver().save(sess, self.modelFile)

    def concatFile(self):
        _data = pd.DataFrame()
        print('Processing Path: ./data/query_log/ \nFile: ')
        for filename in self.csvFiles:
            if self.amount != 0:
                print('------ ' + filename + ' concating...', end='')
                df = pd.read_csv(self.csvFolder + filename, names=self.columns, dtype={'FileID': str, 'CustomerID': str, 'ProductID': str})
                _data = pd.concat([_data, df], axis=0)
                self.amount -= 1
                del df
                print(filename + ' down')
            else:
                break
        _train = pd.read_csv('./data/training-set.csv', names=['FileID', 'VirusRate'], dtype={'FileID': str})
        self.train_data = pd.merge(_data, _train, how='left', on='FileID').values

#for each in sys.argv:
#    if '--amount=' in each:
#        amount = int(each[9:])
#    elif '--train-loop=' in each:
#        trainLoop = int(each[13:])

a = ModelNFMD(amount=2)
a.train(epochs=10)

