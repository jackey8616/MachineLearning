import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

trainBool = True
testBool = False
modelFile = './model_saves/Sklearn/SkLogisticReg.pkl'
logFile = './logs/SkLogisticReg_log_%s.log' % str(time.time())
printer = Printer(logFile)
filer = Filer(printer=printer)

if trainBool:
    _data03 = filer.concatFiles('./data/train_data/_03/')
    _train = pd.read_csv('./data/training-set.csv', names=['FileID', 'VirusRate'], dtype={'FileID': str, 'VirusRate': float})
    excTrain = pd.read_csv('./data/exception/exception_train.txt', names=['FileID'])
    _train = train.loc[pd.merge(train, excTrain, how='left', on='FileID', indicator=True)['_merge'] == 'left_only']
    df = pd.merge(_data03.copy(True), _train, how='left', on='FileID')
    df = df[df.VirusRate.notnull()]
    dfOrigin = df.copy(True)

if testBool:
    _data04 = filer.concatFiles('./data/train_data/_04/')
    #_test = pd.read_csv('./data/testing-set.csv', names=['FileID', 'VirusRate'], dtype={'FileID': str, 'VirusRate': float})
    _train = pd.read_csv('./data/training-set.csv', names=['FileID', 'VirusRate'], dtype={'FileID': str, 'VirusRate': float})
    excTrain = pd.read_csv('./data/exception/exception_train.txt', names=['FileID'])
    _train = train.loc[pd.merge(train, excTrain, how='left', on='FileID', indicator=True)['_merge'] == 'left_only']
    df2 = pd.merge(_data04.copy(True), _train, how='left', on='FileID')
    df2Origin = df2.copy(True)

if trainBool:
    df.CustomerID = LabelEncoder().fit_transform(df.CustomerID)
    df.ProductID = LabelEncoder().fit_transform(df.ProductID)
    #train_X, test_X, train_y, test_y = train_test_split(df.drop(columns=['FileID', 'VirusRate']), df.VirusRate, test_size=0.1)
    train_X = df.drop(columns=['FileID', 'VirusRate'])
    train_y = df.VirusRate

if testBool:
    df2.CustomerID = LabelEncoder().fit_transform(df2.CustomerID)
    df2.ProductID = LabelEncoder().fit_transform(df2.ProductID)
    test_X = df2.drop(columns=['FileID', 'VirusRate'])
    test_y = df2.VirusRate

#logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
logistic = linear_model.LogisticRegression(solver='sag')
logistic = joblib.load(modelFile) if isfile(modelFile) else logistic
printer.Print('Model inited.')
if trainBool:
    printer.Print('Training...')
    logistic_model = logistic.fit(train_X, train_y)
    joblib.dump(logistic, modelFile)
    printer.Print('Model saved.')
    
if testBool:
    logistic_model = logistic
    test_y_predict = logistic_model.predict(test_X)
    test_y_proba = logistic_model.predict_proba(test_X)
    dfVirusRate = pd.DataFrame({'VirusRate': test_y_proba[:,:1].flatten()})
    whole = pd.merge(test_X, dfVirusRate, left_index=True, right_index=True)

    printer.Print('Coef: ', end='')
    printer.Print(logistic_model.coef_, timeInput=False)
    printer.Print('Intercept: ', end='')
    printer.Print(logistic_model.intercept_, timeInput=False)
    printer.Print('Predict: ', end='')
    printer.Print(test_y_predict, timeInput=False)
    printer.Print('')
    printer.Print('Whole:', timeInput=False)
    printer.Print(whole, timeInput=False)

    printer.PrintJudgeResult(test_y, test_y_predict)

printer.Print('Done')
printer.LogClose()

