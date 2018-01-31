from os import listdir
from os.path import isfile, join
import sys
import time, datetime
import tensorflow as tf
import pandas as pd

class TenModelNFMD:

    def __init__(self):
        self.initTime = time.time()
        self.overridePrint('Model initing...')
        self.fileID = tf.placeholder(tf.string, [None, 1])
        self.customerID = tf.placeholder(tf.string, [None, 1])
        self.queryTS = tf.placeholder(tf.float32, [None, 1])
        self.productID = tf.placeholder(tf.string, [None, 1])
        self.y = tf.placeholder(tf.float32, [None, 1])
        
        self.W = tf.Variable(tf.random_normal([1, 1]))
        self.b = tf.Variable(tf.zeros([1, 1]) + 1)
        self.overridePrint('Variable defined.')
   
    def overridePrint(self, msg, end='\n', timeInput=True):
        print(' {}'.format(msg) if timeInput == False else '[{} ({:9.3f} Sec)] {}'.format(str(datetime.datetime.now()), time.time() - self.initTime, msg), end=end)

    def concatFile(self, csvFolder, trainSet, amount=-1):
        columns = ['FileID', 'CustomerID', 'QueryTS', 'ProductID']
        csvFiles = [f for f in listdir(csvFolder) if isfile(join(csvFolder, f))]
        amount = len(csvFiles) if amount == -1 else amount
        count = 0
        errorCount = 0
        self.overridePrint('Processing Path: %s' % csvFolder)
        self.overridePrint('Loading %d files: ' % amount)
        _data = pd.DataFrame()
        for filename in csvFiles:
            if amount != 0:
                try:
                    self.overridePrint('------ (File %d) %s concating...' %  (count + 1, filename), end='')
                    df = pd.read_csv(csvFolder + filename, names=columns, dtype={'FileID': str, 'CustomerID': str, 'ProductID': str})
                    _data = pd.concat([_data, df], axis=0)
                    del df
                    self.overridePrint('%s down.' % filename, timeInput=False)
                    count += 1
                except:
                    self.overridePrint('%s failed.' % filename, timeInput=False)
                    errorCount += 1
                amount -= 1
            else:
                break

        _train = pd.read_csv(trainSet, names=['FileID', 'VirusRate'])
        self.overridePrint('All files (%d files) loaded, %d errors file.' % (count, errorCount))
        self.train_data = pd.merge(_data, _train, how='left', on='FileID').fillna(0.0).values
        self.train_fileID = self.train_data[:, 0:1]
        self.train_customerID = self.train_data[:, 1:2]
        self.train_queryTS = self.train_data[:, 2:3]
        self.train_productID = self.train_data[:, 3:4]
        self.overridePrint('Seperated column from train data.')
        self.train_X = self.train_data[:, :1]
        self.train_y = self.train_data[:, -1:]
        self.overridePrint('Size of train_X: {}x{}'.format(len(self.train_X), len(self.train_X[0])))
        self.overridePrint('Size of train_y: {}x{}'.format(len(self.train_y), len(self.train_y[0])))

    def train(self, sess, msgIntercept=10, epochs=100, rate=0.05):
        self.overridePrint('Training...')
        #z = tf.matmul(X, W) + b
        #o = tf.sigmoid(z)
        #cross_entropy = tf.reduce_mean(y * -tf.log(o) + (1 - y) * -tf.log(1 - o))
        hypo = self.W + self.b
        cross_entropy = tf.reduce_mean(tf.square(hypo - self.y))
        optimizer = tf.train.GradientDescentOptimizer(rate)
        train = optimizer.minimize(cross_entropy)

        optimal_W = None
        optimal_b = None
        for i in range(1, epochs + 1):
            sess.run(train,
                     feed_dict={
                         self.fileID: self.train_fileID,
                         self.customerID: self.train_customerID,
                         self.queryTS: self.train_queryTS,
                         self.productID: self.train_productID,
                         self.y: self.train_y
                         }
                    )
            sys.stdout.write('[{} ({:9.3f} Sec)] '.format(str(datetime.datetime.now()), time.time() - self.initTime))
            for j in range(((epochs - i) % msgIntercept + 1) if i % msgIntercept == 0 else i % msgIntercept):
                sys.stdout.write(' . ') 
            if i % msgIntercept == 0 or i == epochs:
                sys.stdout.write('%d %s\n' % (i, sess.run(cross_entropy,
                                              feed_dict={
                                                 self.fileID: self.train_fileID,
                                                 self.customerID: self.train_customerID,
                                                 self.queryTS: self.train_queryTS,
                                                 self.productID: self.train_productID,
                                                 self.y: self.train_y
                                              }
                                            )
                                        )
                )
            else:
                sys.stdout.write('\r')
            sys.stdout.flush()
        optimal_W = sess.run(self.W)
        optimal_b = sess.run(self.b)
        self.overridePrint('Train finished.')

    def saveModel(self, sess, path):
        try:
            self.overridePrint('Model saving...')
            tf.train.Saver().save(sess, path)
            self.overridePrint('Model saved at %s' % path)
            return True
        except:
            return False

    def restoreModel(self, sess, path):
        try:
            self.overridePrint('Model loading...')
            tf.train.Saver().restore(sess, path)
            self.overridePrint('Model loaded at %s' % path)
            return True
        except:
            return False
