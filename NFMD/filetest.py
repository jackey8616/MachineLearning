from os import listdir
from os.path import join, isfile, isdir

def getFiles(filePath):
    files = []
    for f in listdir(filePath):
        if isfile(join(filePath, f)):
            files.append(f)
        elif isdir(join(filePath, f)):
            files.extend(getFiles(join(filePath, f)))
    return files

print(getFiles('./data/train_data/'))
