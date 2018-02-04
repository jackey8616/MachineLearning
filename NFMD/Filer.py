from os import listdir
from os.path import join, isfile, isdir
import pandas as pd

class Filer:
    
    def __init__(self, printer):
        self.printer = printer
        self.printer.Print('Filer Inited.')
    
    def getFiles(self, filePath):
        files = []
        if(isfile(filePath)):
            return [filePath]
        for f in listdir(filePath):
            if isfile(join(filePath, f)):
                files.append(join(filePath, f))
            elif isdir(join(filePath, f)):
                files.extend(self.getFiles(join(filePath, f)))
        return files
    
    def concatFiles(self, filesPath, dropDupSet=None):
        self.printer.Print('Concating files from path: %s' % filesPath)
        if dropDupSet is not None:
            self.printer.Print('Rows will be droped base on duplicate field check: %s' % str(dropDupSet))
        folder = filesPath
        files = self.getFiles(filesPath)
        colDataNames = ['FileID', 'CustomerID', 'QueryTS', 'ProductID']
        colVirNames = ['FileID', 'VirusRate']
        count = 0
        
        _data = pd.DataFrame()
        for filename in files:
            self.printer.Print('- (File %d) %s concating...' %  (count + 1, filename))
            _read = pd.read_csv(filename, names=colDataNames, dtype={'FileID': str, 'CustomerID': str, 'ProductID': str})
            _data = pd.concat([_data, _read], axis=0)
            if dropDupSet is not None:
                beforeDropLen = len(_data)
                _data = _data.drop_duplicates(subset=dropDupSet)
                self.printer.Print('--- After concate, original length: %d, after drop length: %s.' % (beforeDropLen, len(_data)))
            del _read
            self.printer.Print('- %s done.' % filename)
            count += 1
        self.printer.Print('Files concated.')
        return _data