from src.config.config import config
from src.data.Dataset import Dataset
from src.log.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.getPathFileLog())

import glob
import json
import os
import random
import numpy as np
import torch
from torch.functional import split

class DataManeger(torch.utils.data.Dataset):
    def __init__(self):
        self.pathsSample4Train = []
        self.pathsSample4Test = []
        self.samples4Train = []
        self.samples4Test = []
        self.datasets_Train_Valid = []
        self.datasets_Train_Test = {}
        self.selectedInputData = config.typesInput
    def setPathsSample(self, *, pathsDir, isForTest):
        if(isForTest):
            for path in pathsDir:
                if(os.path.isdir(path)):
                    pathsSearched = glob.glob(path+"/**/*.json", recursive=True)
                    self.pathsSample4Test.extend(pathsSearched)
        else:
            for path in pathsDir:
                if(os.path.isdir(path)):
                    pathsSearched = glob.glob(path+"/**/*.json", recursive=True)
                    self.pathsSample4Train.extend(pathsSearched)
    def loadSamples(self):
        listSamples = []
        if(config.checkMnist4TestExists()):
            sample = {}
            sample["id"] = ""
            sample["y"] = ""
            sample["x"] = {
                "ast": {
                    "nodes": [],
                    "edges": []
                },
                "astseq": [],
                "codemetrics": [],
                "commitgraph": {
                    "nodes": [],
                    "edges": []
                },
                "commitseq": [],
                "processmetrics": []
            }
            listSamples.append(sample)
            random.seed(0)
            random.shuffle(listSamples)
            self.samples4Train = listSamples
        def flattenTree(node):
            seqNode = []
            seqNode.append(node)
            for Child in node["children"]:
                seqNode.extend(flattenTree(Child))
            return seqNode
        def onehotnizeASTNode(numType):
            vectorOnehot = [0] * 99
            vectorOnehot[numType]=1
            return vectorOnehot
        def loadSample(pathSample, listSamples, map2StandardizeMetricsCommit, map2StandardizeMetricsCode, map2StandardizeMetricsProcess):
            sample = {}
            with open(pathSample, encoding="utf-8") as fSample4Train:
                sampleJson = json.load(fSample4Train)
                sample["id"] = sampleJson["path"]
                sample["y"] = sampleJson["commitsOnModuleAll"]["isBuggy"]
                sample["x"] = {
                    "ast": {
                        "nodes": [],
                        "edges": []
                    },
                    "astseq": [],
                    "codemetrics": [],
                    "commitgraph": {
                        "nodes": [],
                        "edges": []
                    },
                    "commitseq": [],
                    "processmetrics": []
                }
                if(config.checkASTExists()):
                    def findNodes(node):
                        nodes = flattenTree(node)
                        for index, node in enumerate(nodes):
                            nodes[index] = onehotnizeASTNode(node["numType"])
                        return nodes
                    def findEdges(node):
                        edges = []
                        nodes = flattenTree(node)
                        for index, node in enumerate(nodes):
                            for child in node["children"]:
                                edges.append([int(node["num"]), int(child["num"])])
                        return edges
                    sample["x"]["ast"]["nodes"] = findNodes(sampleJson["ast"])
                    sample["x"]["ast"]["edges"] = findEdges(sampleJson["ast"])
                if(config.checkASTSeqExists()):
                    def onehotnizeAST(ast):
                        astOnehotnized = []
                        for node in ast:
                            nodeOnehotnized = onehotnizeASTNode(node["numType"])
                            astOnehotnized.append(nodeOnehotnized)
                        return astOnehotnized
                    nodes = flattenTree(sampleJson["ast"])
                    sample["x"]["astseq"] = onehotnizeAST(nodes)
                if(config.checkCodeMetricsExists()):
                    for item in map2StandardizeMetricsCode:
                        sample["x"]["codemetrics"].append((float(sampleJson["sourcecode"][item])-map2StandardizeMetricsCode[item][0]) / map2StandardizeMetricsCode[item][1])
                if(config.checkCommitGraphExists()):
                    pass
                if(config.checkCommitSeqExists()):
                    commitsOnModule = []
                    for commitOnModule in sampleJson["commitsOnModuleInInterval"]["commitsOnModule"].values():
                        commitsOnModule.append(commitOnModule)
                    commitsOnModule = sorted(commitsOnModule, key=lambda x: x['date'])
                    sample["x"]["commitseq"] = []
                    if(0<len(commitsOnModule)):
                        for commitOnModule in commitsOnModule:
                            commitVector = []
                            for nameVector in map2StandardizeMetricsCommit:
                                for indexVector in range(len(map2StandardizeMetricsCommit[nameVector])):
                                    if(map2StandardizeMetricsCommit[nameVector][indexVector][1]!=0):
                                        commitVector.append((commitOnModule[nameVector][indexVector]-map2StandardizeMetricsCommit[nameVector][indexVector][0])/map2StandardizeMetricsCommit[nameVector][indexVector][1])
                                    else:
                                        pass
                            sample["x"]["commitseq"].append(commitVector)
                    else:
                        #コミットがないのはおかしい。飛ばす。
                        return
                if(config.checkProcessMetricsExists()):
                    for item in map2StandardizeMetricsProcess:
                        sample["x"]["processmetrics"].append((float(sampleJson["commitsOnModuleInInterval"][item])-map2StandardizeMetricsProcess[item][0]) / map2StandardizeMetricsProcess[item][1])
            listSamples.append(sample)
        metricsCommit = {
            "vectorSemanticType": [],
            "vectorAuthor": [],
            "vectorInterval": [],
            "vectorType": [],
            "vectorCodeChurn": [],
            "vectorCochange": []
        }
        metricsCode = {
            "fanin" : [],
            "fanout" : [],
            "numOfParameters" : [],
            "numOfVariablesLocal" : [],
            "ratioOfLinesComment" : [],
            "numOfPaths" : [],
            "complexity" : [],
            "numOfStatements" : [],
            "maxOfNesting" : []
        }
        metricsProcess = {
            "numOfCommits" : [],
            "sumOfAdditionsStatement" : [],
            "maxOfAdditionsStatement" : [],
            "avgOfAdditionsStatement" : [],
            "sumOfDeletionsStatement" : [],
            "maxOfDeletionsStatement" : [],
            "avgOfDeletionsStatement" : [],
            "sumOfChurnsStatement" : [],
            "maxOfChurnsStatement" : [],
            "avgOfChurnsStatement" : [],
            "sumOfChangesDeclarationItself" : [],
            "sumOfChangesCondition" : [],
            "sumOfAdditionStatementElse" : [],
            "sumOfDeletionStatementElse" : [],
        }
        map2StandardizeMetricsCommit = {
            "vectorSemanticType": [],
            "vectorAuthor": [],
            "vectorInterval": [],
            "vectorType": [],
            "vectorCodeChurn": [],
            "vectorCochange": []
        }
        map2StandardizeMetricsCode = {
            "fanin" : [],
            "fanout" : [],
            "numOfParameters" : [],
            "numOfVariablesLocal" : [],
            "ratioOfLinesComment" : [],
            "numOfPaths" : [],
            "complexity" : [],
            "numOfStatements" : [],
            "maxOfNesting" : []
        }
        map2StandardizeMetricsProcess = {
            "numOfCommits" : [],
            "sumOfAdditionsStatement" : [],
            "maxOfAdditionsStatement" : [],
            "avgOfAdditionsStatement" : [],
            "sumOfDeletionsStatement" : [],
            "maxOfDeletionsStatement" : [],
            "avgOfDeletionsStatement" : [],
            "sumOfChurnsStatement" : [],
            "maxOfChurnsStatement" : [],
            "avgOfChurnsStatement" : [],
            "sumOfChangesDeclarationItself" : [],
            "sumOfChangesCondition" : [],
            "sumOfAdditionStatementElse" : [],
            "sumOfDeletionStatementElse" : [],
        }
        if(config.checkCommitSeqExists()):
            # vectorのサイズを決定
            with open(self.pathsSample4Train[0], encoding="utf-8") as fSample4Train:
                sampleJson = json.load(fSample4Train)
                for commitOnModule in sampleJson["commitsOnModuleInInterval"]["commitsOnModule"].values():
                    metricsCommit["vectorSemanticType"] = [[] for x in range(len(commitOnModule["vectorSemanticType"]))]
                    metricsCommit["vectorAuthor"] = [[] for x in range(len(commitOnModule["vectorAuthor"]))]
                    metricsCommit["vectorInterval"] = [[] for x in range(len(commitOnModule["vectorInterval"]))]
                    metricsCommit["vectorType"] = [[] for x in range(len(commitOnModule["vectorType"]))]
                    metricsCommit["vectorCodeChurn"] = [[] for x in range(len(commitOnModule["vectorCodeChurn"]))]
                    metricsCommit["vectorCochange"] = [[] for x in range(len(commitOnModule["vectorCochange"]))]
                    map2StandardizeMetricsCommit["vectorSemanticType"] = [[] for x in range(len(commitOnModule["vectorSemanticType"]))]
                    map2StandardizeMetricsCommit["vectorAuthor"] = [[] for x in range(len(commitOnModule["vectorAuthor"]))]
                    map2StandardizeMetricsCommit["vectorInterval"] = [[] for x in range(len(commitOnModule["vectorInterval"]))]
                    map2StandardizeMetricsCommit["vectorType"] = [[] for x in range(len(commitOnModule["vectorType"]))]
                    map2StandardizeMetricsCommit["vectorCodeChurn"] = [[] for x in range(len(commitOnModule["vectorCodeChurn"]))]
                    map2StandardizeMetricsCommit["vectorCochange"] = [[] for x in range(len(commitOnModule["vectorCochange"]))]
                    break
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    for commitOnModule in sampleJson["commitsOnModuleInInterval"]["commitsOnModule"].values():
                        for item in metricsCommit:
                            for i in range(len(metricsCommit[item])):
                                metricsCommit[item][i].append(commitOnModule[item][i])
            for item in map2StandardizeMetricsCommit:
                for i in range(len(metricsCommit[item])):
                    map2StandardizeMetricsCommit[item][i] = [np.array(metricsCommit[item][i]).mean(), np.std(metricsCommit[item][i])]
            del metricsCommit
        if(config.checkCodeMetricsExists()):
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    for item in metricsCode:
                        metricsCode[item].append(sampleJson["sourcecode"][item])
            for item in map2StandardizeMetricsCode:
                map2StandardizeMetricsCode[item] = [np.array(metricsCode[item]).mean(), np.std(metricsCode[item])]
        if(config.checkProcessMetricsExists()):
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    for item in metricsProcess:
                        metricsProcess[item].append(float(sampleJson["commitsOnModuleInInterval"][item]))
            for item in map2StandardizeMetricsProcess:
                map2StandardizeMetricsProcess[item] = [np.array(metricsProcess[item]).mean(), np.std(metricsProcess[item])]
        for pathSample4Train in self.pathsSample4Train:
            loadSample(pathSample4Train, self.samples4Train, map2StandardizeMetricsCommit, map2StandardizeMetricsCode, map2StandardizeMetricsProcess)
        for pathSample4Test in self.pathsSample4Test:
            loadSample(pathSample4Test, self.samples4Test, map2StandardizeMetricsCommit, map2StandardizeMetricsCode, map2StandardizeMetricsProcess)
        logger.info( "\n" +
            "samples4TrainPositive: "+ str(self.getNumOfSamples4TrainPositive()) + "\n" +
            "samples4TrainNegative: "+ str(self.getNumOfSamples4TrainNegative()) + "\n" +
            "samples4TestPositive: "+ str(self.getNumOfSamples4TestPositive()) + "\n" +
            "samples4TestNegative: "+ str(self.getNumOfSamples4TestNegative())
        )
    def generateDatasetsTrainValid(self, *, isCrossValidation, numOfSplit):
        samplesBuggy    = []
        samplesNotBuggy = []
        for data in self.samples4Train:
            if(data["y"]==1):
                samplesBuggy.append(data)
            elif(int(data["y"])==0):
                samplesNotBuggy.append(data)
        random.seed(0)
        random.shuffle(samplesBuggy)
        random.shuffle(samplesNotBuggy)
        for i in range(numOfSplit):
            dataset_train_valid = {}
            dataset4Train=[]
            dataset4Valid=[]
            validBuggy = samplesBuggy[(len(samplesBuggy)//numOfSplit)*i:(len(samplesBuggy)//numOfSplit)*(i+1)]
            validNotBuggy = samplesNotBuggy[(len(samplesNotBuggy)//numOfSplit)*i:(len(samplesNotBuggy)//numOfSplit)*(i+1)]
            validBuggy = random.choices(validBuggy, k=len(validNotBuggy))
            dataset4Valid.extend(validBuggy)
            dataset4Valid.extend(validNotBuggy)
            random.shuffle(dataset4Valid)#最初に1, 次に0ばっかり並んでしまっている。

            trainBuggy = samplesBuggy[:(len(samplesBuggy)//numOfSplit)*i]+samplesBuggy[(len(samplesBuggy)//numOfSplit)*(i+1):]
            trainNotBuggy = samplesNotBuggy[:(len(samplesNotBuggy)//numOfSplit)*i]+samplesNotBuggy[(len(samplesNotBuggy)//numOfSplit)*(i+1):]
            trainBuggy = random.choices(trainBuggy, k=len(trainNotBuggy))
            dataset4Train.extend(trainBuggy)
            dataset4Train.extend(trainNotBuggy)
            random.shuffle(dataset4Train)#最初に1, 次に0ばっかり並んでしまっている。
            dataset_train_valid["train"] = dataset4Train
            dataset_train_valid["valid"] = dataset4Valid
            self.datasets_Train_Valid.append(dataset_train_valid)
            if(not isCrossValidation):
                break
    def generateDatasetsTrainTest(self):
        dataset_Train_Test={}
        dataset4Train=[]
        dataset4Test=[]
        samplesBuggy    = []
        samplesNotBuggy = []
        for data in self.samples4Train:
            if(data['y']==1):
                samplesBuggy.append(data)
            elif(data['y']==0):
                samplesNotBuggy.append(data)
        random.seed(0)
        samplesBuggy = random.choices(samplesBuggy, k=len(samplesNotBuggy))
        dataset4Train.extend(samplesBuggy)
        dataset4Train.extend(samplesNotBuggy)
        random.shuffle(dataset4Train)#最初に1, 次に0ばっかり並んでしまうのを防ぐ。
        dataset4Test = self.samples4Test
        dataset_Train_Test["train"] = dataset4Train
        dataset_Train_Test["test"] = dataset4Test
        self.datasets_Train_Test = dataset_Train_Test
    def getNumOfSamples4TrainPositive(self):
        return len([sample for sample in self.samples4Train if sample["y"]==1])
    def getNumOfSamples4TrainNegative(self):
        return len([sample for sample in self.samples4Train if sample["y"]==0])
    def getNumOfSamples4TestPositive(self):
        return len([sample for sample in self.samples4Test if sample["y"]==1])
    def getNumOfSamples4TestNegative(self):
        return len([sample for sample in self.samples4Test if sample["y"]==0])