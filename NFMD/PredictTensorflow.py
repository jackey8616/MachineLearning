import tensorflow as tf
import pandas as pd

data = pd.read_csv('./data/query_log/0501.csv', names=['FileID'], dtype={'FileID': str})

with tf.Session() as sess:
    W = tf.Variable(tf.random_normal([1, 1]))
    b = tf.Variable(tf.zeros([1, 1]) + 1)

    fileID = tf.placeholder(tf.string, [None, 1])
    #y = tf.placeholder(tf.float32, [None, 1])
    y = tf.nn.softmax(W + b)

    saver = tf.train.Saver()
    saver.restore(sess, './model_saves/Tensorflow/TenLogisticReg.ckpt')
    result = sess.run(y, feed_dict={fileID: data})

    for i in result:
        print(i)

