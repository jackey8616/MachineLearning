from os import listdir
from os.path import isfile, join
import sys
import tensorflow as tf
import pandas as pd

class ModelNFMD:

    def __init__(self):
        print('Model initing...') 
        self.fileID = tf.placeholder(tf.string, [None, 1])
        self.customerID = tf.placeholder(tf.string, [None, 1])
        self.queryTS = tf.placeholder(tf.float32, [None, 1])
        self.productID = tf.placeholder(tf.string, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        
        self.W = tf.Variable(tf.random_normal([1, 1]))
        self.b = tf.Variable(tf.zeros([1, 1]) + 1)
        print('Variable defined.')
   

    def concatFile(self, amount=10):
        columns = ['FileID', 'CustomerID', 'QueryTS', 'ProductID']
        csvFolder = './data/train_data/'
        csvFiles = [f for f in listdir(csvFolder) if isfile(join(csvFolder, f))]
        _data = pd.DataFrame()
        print('Processing Path: ./data/query_log/ \nFile: ')
        for filename in csvFiles:
            if amount != 0:
                print('------ ' + filename + ' concating...', end='')
                df = pd.read_csv(csvFolder + filename, names=columns, dtype={'FileID': str, 'CustomerID': str, 'ProductID': str})
                _data = pd.concat([_data, df], axis=0)
                amount -= 1
                del df
                print(filename + ' down')
            else:
                break
        _train = pd.read_csv('./data/training-set.csv', names=['FileID', 'VirusRate'], dtype={'FileID': str})
        self.train_data = pd.merge(_data, _train, how='left', on='FileID').values
        self.train_fileID = self.train_data[:, 0:1]
        self.train_customerID = self.train_data[:, 1:2]
        self.train_queryTS = self.train_data[:, 2:3]
        self.train_productID = self.train_data[:, 3:4]
        print('Seperated column from train data.')
        self.train_X = self.train_data[:, :1]
        self.train_y = self.train_data[:, -1:]
        print('Size of train_X: {}x{}'.format(len(self.train_X), len(self.train_X[0])))
        print('Size of train_y: {}x{}'.format(len(self.train_y), len(self.train_y[0])))

    def train(self, sess, epochs=100, rate=0.05):
        print('Training...')
        #z = tf.matmul(X, W) + b
        #o = tf.sigmoid(z)
        #cross_entropy = tf.reduce_mean(y * -tf.log(o) + (1 - y) * -tf.log(1 - o))
        hypo = self.W + self.b
        cross_entropy = tf.reduce_mean(tf.square(hypo - self.y))
        optimizer = tf.train.GradientDescentOptimizer(rate)
        train = optimizer.minimize(cross_entropy)

        optimal_W = None
        optimal_b = None
        for i in range(epochs):
            sess.run(train,
                     feed_dict={
                         self.fileID: self.train_fileID,
                         self.customerID: self.train_customerID,
                         self.queryTS: self.train_queryTS,
                         self.productID: self.train_productID,
                         self.y: self.train_y
                         }
                    )
            #if i % 50 == 0:
            print(i, sess.run(cross_entropy,
                              feed_dict={
                                  self.fileID: self.train_fileID,
                                  self.customerID: self.train_customerID,
                                  self.queryTS: self.train_queryTS,
                                  self.productID: self.train_productID,
                                  self.y: self.train_y
                                  }
                             )
                 )
            #print(self.train_data)
        optimal_W = sess.run(self.W)
        optimal_b = sess.run(self.b)

    def saveModel(self, sess, path):
        tf.train.Saver().save(sess, path)

#for each in sys.argv:
#    if '--amount=' in each:
#        amount = int(each[9:])
#    elif '--train-loop=' in each:
#        trainLoop = int(each[13:])

modelFile = './model_saves/Tensorflow/TenLogisticReg.ckpt'
a = ModelNFMD()
a.concatFile(amount=2)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    a.train(sess, epochs=10)
    a.saveModel(sess, modelFile)

