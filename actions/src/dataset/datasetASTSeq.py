import ast
import os
import csv
import copy
import random

import numpy as np
import glob
import json

class DatasetASTSeq():
    def __init__(self):
        self.pathsSample = []
        self.samples4Train = []
        self.samples4Test = []
        self.splitSize4Validation = 5
        self.isCrossValidation = True

    def setSplitSize4Validation(self, splitSize4Validation):
        self.splitSize4Validation = splitSize4Validation

    def setIsCrossValidation(self, isCrossValidation):
        self.isCrossValidation = isCrossValidation

    def setPathSamples(self, pathsSample):
        self.pathsSample = pathsSample

    def flattenAST(self, ASTNode):
        NodesAST = []
        NodesAST.append(ASTNode)
        for Child in ASTNode["children"]:
            NodesAST.extend(self.flattenAST(Child))
        return NodesAST

    def loadDataset(self):
        def onehotnizeASTNode(numType):
            vectorOnehot = [0] * 99
            vectorOnehot[numType]=1
            return vectorOnehot
        def onehotnizeAST(ast):
            astOnehotnized = []
            for node in ast:
                nodeOnehotnized = onehotnizeASTNode(node["numType"])
                astOnehotnized.append(nodeOnehotnized)
            return astOnehotnized
        samples = []
        for pathRoot in self.pathsSample:
            pathsSample = glob.glob(pathRoot+"\\**\\*.json", recursive=True)
            for pathSample in pathsSample:
                with open(pathSample, encoding="utf-8") as f:
                    sample = json.load(f)
                    ast = [[0] * 99] * 1024
                    astOnehotnized = onehotnizeAST(self.flattenAST(sample["ast"]))
                    if(1024<len(astOnehotnized)):
                        print(pathSample)
                        print(sample["isBuggy"])
                    else:
                        for i, Node in enumerate(astOnehotnized):
                            ast[i] = Node
                        samples.append([sample["path"], sample["isBuggy"], ast])
        print(len(samples))
        random.seed(0)
        for i in range(len(samples)//5):
            index = random.randint(0, len(sample)-1)
            self.samples4Test.append(samples.pop(index))
        self.samples4Train = samples


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
        for data in self.samples4Train:
            if(int(data[1])==1):
                recordsBuggy.append(data)
            elif(int(data[1])==0):
                recordsNotBuggy.append(data)
        for i in range(self.splitSize4Validation):
            dataset = {}
            dataset4Train=[]
            dataset4Valid=[]
            validBuggy = recordsBuggy[(len(recordsBuggy)//5)*i:(len(recordsBuggy)//5)*(i+1)]
            validNotBuggy = recordsNotBuggy[(len(recordsNotBuggy)//5)*i:(len(recordsNotBuggy)//5)*(i+1)]
            validBuggy = random.choices(validBuggy, k=len(validNotBuggy))
            dataset4Valid.extend(validBuggy)
            dataset4Valid.extend(validNotBuggy)
            random.shuffle(dataset4Valid)#最初に1, 次に0ばっかり並んでしまっている。

            trainBuggy = recordsBuggy[:(len(recordsBuggy)//5)*i]+recordsBuggy[(len(recordsBuggy)//5)*(i+1):]
            trainNotBuggy = recordsNotBuggy[:(len(recordsNotBuggy)//5)*i]+recordsNotBuggy[(len(recordsNotBuggy)//5)*(i+1):]
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
            return [arrayOfD4TAndD4V[0]]

    def getDataset4SearchParameter(self):
        dataset={}
        dataset4Train=[]
        dataset4Valid=[]
        samplesBuggy    = []
        samplesNotBuggy = []
        for data in self.samples4Train:
            if(data[1]==1):
                samplesBuggy.append(data)
            elif(data[1]==0):
                samplesNotBuggy.append(data)
        samplesNotBuggy = random.sample(samplesNotBuggy, k=len(samplesBuggy))
        #samplesBuggy = random.choices(samplesBuggy, k=len(samplesNotBuggy))
        dataset4Train.extend(samplesBuggy)
        dataset4Train.extend(samplesNotBuggy)
        random.shuffle(dataset4Train)#最初に1, 次に0ばっかり並んでしまうのを防ぐ。
        dataset4Valid = self.samples4Test
        dataset["train"] = dataset4Train
        dataset["valid"] = dataset4Valid
        return dataset

    def getDataset4Test(self):
        return self.samples4Test

    def showSummary(self):
        print(" pathsSample: ")
        print(self.pathsSample)
        print(" len(samples4Train): " + str(len(self.samples4Train)))
        print(" len(samples4Test): " + str(len(self.samples4Test)))