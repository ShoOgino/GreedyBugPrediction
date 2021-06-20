from src.utility import UtilPath
import numpy as np
import os
import csv

class DatasetLoader:
    def __init__(self, pathDatasetDir, purpose):
        self.pathDatasetDir = pathDatasetDir
        self.porpose = purpose

    def loadTrain4Search(self, indexCV=0):
        dataTrain=[[],[],[]]
        pathDatasetTrain = os.path.join(UtilPath.Datasets(),self.pathDatasetDir,"train"+str(indexCV)+".csv")
        with open(pathDatasetTrain) as f:
            train = csv.reader(f)
            for i, row in enumerate(train):
                dataTrain[0].append(row[0])
                dataTrain[1].append([float(x) for x in row[2:]])
                dataTrain[2].append(int(row[1]))
            dataTrain[0]=np.array(dataTrain[0])
            dataTrain[1]=np.array(dataTrain[1])
            dataTrain[2]=np.array(dataTrain[2])
            return dataTrain[0], dataTrain[1], dataTrain[2]

    def loadValid4Search(self, indexCV=0):
        dataValid=[[],[],[]]
        pathDatasetValid = os.path.join(UtilPath.Datasets(), self.pathDatasetDir, "valid"+str(indexCV)+".csv")
        with open(pathDatasetValid) as f:
            valid = csv.reader(f)
            for i, row in enumerate(valid):
                dataValid[0].append(row[0])
                dataValid[1].append([float(x) for x in row[2:]])
                dataValid[2].append(int(row[1]))
            dataValid[0] = np.array(dataValid[0])
            dataValid[1] = np.array(dataValid[1])
            dataValid[2] = np.array(dataValid[2])
        return dataValid[0], dataValid[1], dataValid[2]

    def loadTrain4Test(self, indexCV=0):
        dataTrain=[[],[],[]]
        dataValid=[[],[],[]]
        pathDatasetTrain=os.path.join(UtilPath.Datasets(), self.pathDatasetDir, "train"+str(indexCV)+".csv")
        with open(pathDatasetTrain) as f:
            train = csv.reader(f)
            for i, row in enumerate(train):
                dataTrain[0].append(row[0])
                dataTrain[1].append([float(x) for x in row[2:]])
                dataTrain[2].append(int(row[1]))
        dataTrain[0] = np.array(dataTrain[0])
        dataTrain[1] = np.array(dataTrain[1])
        dataTrain[2] = np.array(dataTrain[2])
        pathDatasetValid = os.path.join(UtilPath.Datasets(),self.pathDatasetDir,"valid"+str(indexCV)+".csv")
        with open(pathDatasetValid) as f:
            valid = csv.reader(f)
            for i, row in enumerate(valid):
                dataValid[0].append(row[0])
                dataValid[1].append([float(x) for x in row[2:]])
                dataValid[2].append(int(row[1]))
        dataValid[0] = np.array(dataValid[0])
        dataValid[1] = np.array(dataValid[1])
        dataValid[2] = np.array(dataValid[2])
        return np.concatenate([dataTrain[0], dataValid[0]]), np.concatenate([dataTrain[1], dataValid[1]]), np.concatenate([dataTrain[2], dataValid[2]], 0)

    def loadTest4Test(self, indexCV=0):
        dataTest=[[],[],[]]
        pathDatasetTest = os.path.join(UtilPath.Datasets(), self.pathDatasetDir, "test.csv")
        with open(pathDatasetTest) as f:
            test = csv.reader(f)
            for i, row in enumerate(test):
                dataTest[0].append(row[0])
                dataTest[1].append([float(x) for x in row[2:]])
                dataTest[2].append(int(row[1]))
        dataTest[0] = np.array(dataTest[0])
        dataTest[1] = np.array(dataTest[1])
        dataTest[2] = np.array(dataTest[2])
        return dataTest[0], dataTest[1], dataTest[2]