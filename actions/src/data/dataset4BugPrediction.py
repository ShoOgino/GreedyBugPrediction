import os
import csv
import copy
import random
import numpy as np

class Dataset4BugPrediction():
    def __init__(self):
        self.records4Train = []
        self.pathsRecords4Train = []
        self.records4Test = []
        self.pathsRecords4Test = []
        self.splitSize4Validation = 5
        self.isCrossValidation = True

    def loadRecords4Train(self):
        self.records4Train=[]
        for path in self.pathsRecords4Train:
            with open(path, encoding="utf-8") as f:
                records = csv.reader(f)
                for i, row in enumerate(records):
                    if(not ("test" in row[0] or "Test" in row[0])):
                    #if(not (row[11] == "0" or "test" in row[0] or "Test" in row[0])):
                        self.records4Train.append([row[0], int(row[1]), [float(x) for x in row[2:]]])

    def loadRecords4Test(self):
        self.records4Test=[]
        for path in self.pathsRecords4Test:
            with open(path, encoding="utf-8") as f:
                records = csv.reader(f)
                for i, row in enumerate(records):
                    if(not ("test" in row[0] or "Test" in row[0])):
                        self.records4Test.append([row[0], int(row[1]), [float(x) for x in row[2:]]])

    def setSplitSize4Validation(self, splitSize4Validation):
        self.splitSize4Validation = splitSize4Validation

    def setIsCrossValidation(self, isCrossValidation):
        self.isCrossValidation = isCrossValidation

    def setPathsRecords4Train(self, pathsRecords4Train):
        self.pathsRecords4Train = pathsRecords4Train

    def setPathsRecords4Test(self, pathsRecords4Test):
        self.pathsRecords4Test = pathsRecords4Test

    def standardize(self):
        features=[[] for i in range(24)]
        for index in range (24):
            for row in self.records4Train:
                features[index].append(float(row[2][index]))
            mean=np.array(features[index]).mean()
            std=np.std(features[index])
            for row in self.records4Train:
                if(not std == 0):
                    row[2][index]=(float(row[2][index])-mean)/std
                else:
                    pass
            for row in self.records4Test:
                if(not std == 0):
                    row[2][index]=(float(row[2][index])-mean)/std
                else:
                    pass

    def getDataset4SearchHyperParameter(self):
        arrayOfD4TAndD4V = []
        recordsBuggy    = []
        recordsNotBuggy = []
        for data in self.records4Train:
            if(int(data[1])==1):
                recordsBuggy.append(data)
            elif(int(data[1])==0):
                recordsNotBuggy.append(data)
        for i in range(self.splitSize4Validation):
            dataset = {}
            dataset4Train=[]
            dataset4Valid=[]
            validBuggy = copy.deepcopy(recordsBuggy[(len(recordsBuggy)//5)*i:(len(recordsBuggy)//5)*(i+1)])
            validNotBuggy = copy.deepcopy(recordsNotBuggy[(len(recordsNotBuggy)//5)*i:(len(recordsNotBuggy)//5)*(i+1)])
            validBuggy = random.choices(validBuggy, k=len(validNotBuggy))
            dataset4Valid.extend(validBuggy)
            dataset4Valid.extend(validNotBuggy)
            random.shuffle(dataset4Valid)#最初に1, 次に0ばっかり並んでしまっている。

            trainBuggy = copy.deepcopy(recordsBuggy[:(len(recordsBuggy)//5)*i]+recordsBuggy[(len(recordsBuggy)//5)*(i+1):])
            trainNotBuggy = copy.deepcopy(recordsNotBuggy[:(len(recordsNotBuggy)//5)*i]+recordsNotBuggy[(len(recordsNotBuggy)//5)*(i+1):])
            trainBuggy = random.choices(trainBuggy, k=len(trainNotBuggy))
            dataset4Train.extend(trainBuggy)
            dataset4Train.extend(trainNotBuggy)
            random.shuffle(dataset4Train)#最初に1, 次に0ばっかり並んでしまっている。
            dataset["training"] = dataset4Train
            dataset["validation"] = dataset4Valid
            arrayOfD4TAndD4V.append(dataset)
        if(self.isCrossValidation):
            return arrayOfD4TAndD4V
        else:
            return arrayOfD4TAndD4V[0]

    def getDataset4SearchParameter(self):
        dataset=[]
        recordsBuggy    = []
        recordsNotBuggy = []
        for data in self.records4Train:
            if(data[1]==1):
                recordsBuggy.append(data)
            elif(data[1]==0):
                recordsNotBuggy.append(data)
        recordsBuggy = random.choices(recordsBuggy, k=len(recordsNotBuggy))
        dataset.extend(recordsBuggy)
        dataset.extend(recordsNotBuggy)
        random.shuffle(dataset)#最初に1, 次に0ばっかり並んでしまっている。
        return dataset

    def getDataset4Test(self):
        return self.records4Test

    def showSummary(self):
        print(" pathDataset4Train: ")
        print(self.pathsRecords4Train)
        print("len(dataset4Train): " + str(len(self.records4Train)))
        print("  pathDataset4Test: ")
        print(self.pathsRecords4Test)
        print(" len(dataset4Test): " + str(len(self.records4Test)))

