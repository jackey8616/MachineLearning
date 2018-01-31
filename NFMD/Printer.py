import time, datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Printer:
    
    def __init__(self, logFile, initTime=time.time()):
        self.initTime = initTime
        self.log = self.LogInit(logFile) if logFile is not None else None
        self.Print('Printer Inited')
    
    def LogInit(self, filename, mode='w+'):
        return open(filename, mode)
    
    def LogClose(self,):
        self.log.close()
    
    def Print(self, msg, end='\n', timeInput=True, log=True):
        output = ' {}'.format(msg) if timeInput == False else '[{} ({:9.3f} Sec)] {}'.format(str(datetime.datetime.now()), time.time() - self.initTime, msg)
        print(output, end=end)
        if log:
            self.log.write(output + end)
        
    def PrintJudgeResult(self, true, predict):
        accuracy = metrics.accuracy_score(true, predict)
        self.Print('Accuracy: ', end='')
        self.Print(accuracy, timeInput=False)
        accuracyNor = metrics.accuracy_score(true, predict, normalize=False)
        self.Print('Accuracy(Sameples):', end='')
        self.Print(accuracyNor, timeInput=False)

        fpr, tpr, thresholds = metrics.roc_curve(true, predict)
        self.Print('ROC curve:')
        self.Print('-- TPR = TP / (TP + FN): ', end='')
        self.Print(tpr, timeInput=False)
        self.Print('-- FPR = FP / (FP + TN): ', end='')
        self.Print(fpr, timeInput=False)
        self.Print('-- Thresholds: ', end='')
        self.Print(thresholds, timeInput=False)

        auc = metrics.auc(fpr, tpr)
        self.Print('AUC: ', end='')
        self.Print(auc, timeInput=False)
        
    def DrawScatter(self, X, Y, Z, C, xlabel, ylabel, zlabel, figsize=(8, 6), dpi=400):
        fig = plg.figure(figsize=figsize, dpi=dpi)
        ax = Axes3D(fig)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.scatter(X, Y, Z, C)
        plt.show()
