from os import listdir
from os.path import isfile, join
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.externals import joblib

label_encoder = preprocessing.LabelEncoder()
logistic_regr = linear_model.LogisticRegression()

pklFile = 'LogisticReg.pkl'
columns = ['FileID', 'CustomerID', 'QueryTS', 'ProductID']
csvFolder = './data/query_log/'
csvFiles = [f for f in listdir(csvFolder) if isfile(join(csvFolder, f))]
amount = 10

for each in sys.argv:
    if '--pkl=' in each:
        pklFile = each[6:] if os.path.isfile(each[6:]) else pklFile
    elif '--start-file-name=' in each:
        month = int(each[18:])
    elif '--amount=' in each:
        amount = int(each[9:])

# Short test data only 03/01
#_0301 = pd.read_csv('./data/query_log/0301.csv', names=columns)
#_train = pd.read_csv('./data/training-set.csv', names=['FileID', 'VirusRate'])

_data = pd.DataFrame()
print('Processing Path: ./data/query_log/ \nFile: ')
for filename in csvFiles:
    if amount != 0:
        print('------ ' + filename + ' concating...', end='')
        df = pd.read_csv(csvFolder + '/' + filename, names=columns, dtype={'FileID': str, 'CustomerID': str, 'ProductID': str})
        _data = pd.concat([_data, df], axis=0)
        amount -= 1
        del df
        print(filename + ' down')
    else:
        print('Training...')
        _train = pd.read_csv('./data/training-set.csv', names=['FileID', 'VirusRate'])
        df = pd.merge(_data, _train, how='left', on='FileID')

        # Dummy variable
        print('Creating dummy variables')
        encoded_fileID = label_encoder.fit_transform(df['FileID'])
        encoded_customerID = label_encoder.fit_transform(df['CustomerID'])
        encoded_productID = label_encoder.fit_transform(df['ProductID'])
    
        print('Creating train_X data')
        train_X = pd.DataFrame([encoded_fileID, encoded_customerID, df['QueryTS'], encoded_productID]).T
 
        print('Creating Logistic Regression Instance')
        logistic_regr = joblib.load(pklFile) if isfile(pklFile) else logistic_regr
        logistic_regr.fit(train_X, df['VirusRate'])
        joblib.dump(logistic_regr, pklFile)

        print('Coefficient: ', logistic_regr.coef_)
        print('Intercept: ', logistic_regr.intercept_)
        break

print('Predicting')
virusRate_predictions = logistic_regr.predict(train_X)
accuracy = logistic_regr.score(train_X, df['VirusRate'])
print('Accuracy: %f' % accuracy)
