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

def getFiles(filePath):
    files = []
    for f in listdir(filePath):
        if isfile(filePath, f):
            files.append(f)
        elif isfolder(filePath):
            files.append(getFile(filePath))
    return files            

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


model = TenModelNFMD()
model.concatFile(amount=amount, csvFolder=trainDataPath, trainSet=trainSetFile)
with tf.Session() as sess:
    if model.restoreModel(sess, modelLoadPath) == False:
        init = tf.global_variables_initializer()
        sess.run(init)
    model.train(sess, msgIntercept=msgIntercept, epochs=epochs)
    model.saveModel(sess, modelSavePath)

