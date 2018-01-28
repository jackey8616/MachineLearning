import os
import sys
import tensorflow as tf
from TenModelNFMD import TenModelNFMD

amount = -1
epochs = 100
trainDataPath = './data/train_data/'
trainSetFile = './data/training-set.csv'
modelLoadPath = './model_saves/Tensorflow/TenLogisticReg.ckpt'
modelSavePath = modelLoadPath
msgIntercept = 10
TF_CPP_MIN_LOG_LEVEL = os.environ.get('TF_CPP_MIN_LOG_LEVEL')

for each in sys.argv:
    if '--amount=' in each:
        amount = int(each[9:])
    elif '--epochs=' in each:
        epochs = int(each[9:])
    elif '--train-data-path=' in each:
        trainDataPath = str(each[18:])
    elif '--train-set-path=' in each:
        trainSetFile = str(each[17:])
    elif '--model-load=' in each:
        modelLoadPath = str(each[15:])
    elif '--model-save=' in each:
        modelSavePath = str(each[15:])
    elif '--msg-intercept=' in each:
        msgIntercept = int(each[16:])
    elif '--TF_CPP_MIN_LOG_LEVEL=' in each:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(each[23:])

model = TenModelNFMD()
model.concatFile(amount=amount, csvFolder=trainDataPath, trainSet=trainSetFile)
with tf.Session() as sess:
    if model.restoreModel(sess, modelLoadPath) == False:
        init = tf.global_variables_initializer()
        sess.run(init)
    model.train(sess, msgIntercept=msgIntercept, epochs=epochs)
    model.saveModel(sess, modelSavePath)


