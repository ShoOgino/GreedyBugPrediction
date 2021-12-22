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
                    sample["x"]["codemetrics"].extend(
                        [
                            (float(sampleJson["sourcecode"]["fanIn"])-map2StandardizeMetricsCode["fanIn"][0]) / map2StandardizeMetricsCode["fanIn"][1],
                            (float(sampleJson["sourcecode"]["fanOut"])-map2StandardizeMetricsCode["fanOut"][0]) / map2StandardizeMetricsCode["fanOut"][1],
                            (float(sampleJson["sourcecode"]["parameters"])-map2StandardizeMetricsCode["parameters"][0]) / map2StandardizeMetricsCode["parameters"][1],
                            (float(sampleJson["sourcecode"]["localVar"])-map2StandardizeMetricsCode["localVar"][0]) / map2StandardizeMetricsCode["localVar"][1],
                            (float(sampleJson["sourcecode"]["commentRatio"])-map2StandardizeMetricsCode["commentRatio"][0]) / map2StandardizeMetricsCode["commentRatio"][1],
                            (float(sampleJson["sourcecode"]["countPath"])-map2StandardizeMetricsCode["countPath"][0]) / map2StandardizeMetricsCode["countPath"][1],
                            (float(sampleJson["sourcecode"]["complexity"])-map2StandardizeMetricsCode["complexity"][0]) / map2StandardizeMetricsCode["complexity"][1],
                            (float(sampleJson["sourcecode"]["execStmt"])-map2StandardizeMetricsCode["execStmt"][0]) / map2StandardizeMetricsCode["execStmt"][1],
                            (float(sampleJson["sourcecode"]["maxNesting"])-map2StandardizeMetricsCode["maxNesting"][0]) / map2StandardizeMetricsCode["maxNesting"][1]
                        ]
                    )
                if(config.checkCommitGraphExists()):
                    numOfCommitsOnModule = len(sampleJson["commitGraph"])
                    sample["x"]["commitgraph"]["nodes"] = [[None] for i in range(numOfCommitsOnModule)]
                    for i in range(numOfCommitsOnModule):
                        node = sampleJson["commitGraph"][i]
                        vector = [
                            [],
                            [0] * len(committer2Num),
                            [0] * 2,
                            [0] * 50,
                            [[0] * 50 for i in range(3)],
                            [0] * len(module2Num)
                        ]
                        vector[0] = node["semantics"]
                        if(node["author"] in committer2Num):
                            vector[1][committer2Num[node["author"]]] = 1
                        else:
                            vector[1][0] = 1
                        if(node["isMerge"]==True):
                            vector[2][0] = 1
                        else:
                            vector[2][0] = 0
                        if(node["isFixingBug"]==True):
                            vector[2][1] = 1
                        else:
                            vector[2][1] = 0
                        for i in range (50):
                            if(node["interval"] <= tableInterval[i]):
                                vector[3][i]=1
                                break
                        for type in range(3): #add, delete, churn
                            for i in range(50):
                                if( node["churn"][type] <= tableChurn[type][i] ):
                                    vector[4][type][i] = 1
                                    break
                        for pathModule in node["coupling"]:
                            if(pathModule in module2Num):
                                vector[5][module2Num[pathModule]] = 1
                            else:
                                vector[5][0] = 1
                        sample["x"]["commitgraph"]["nodes"][node["num"]] = vector[0]+vector[1]+vector[2]+vector[3]+vector[4][0]+vector[4][1]+vector[4][2]+vector[5]
                        for numParent in node["parents"]:
                            sample["x"]["commitgraph"]["edges"].append([node["num"], numParent])
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
                                        commitVector.append(0)
                            sample["x"]["commitseq"].append(commitVector)
                    else:
                        length = 0
                        for nameVector in map2StandardizeMetricsCommit:
                            length += len(map2StandardizeMetricsCommit[nameVector])
                        commitVector=[0 for x in range(length)]
                        sample["x"]["commitseq"].append(commitVector)
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
            "fanIn" : [],
            "fanOut" : [],
            "parameters" : [],
            "localVar" : [],
            "commentRatio" : [],
            "countPath" : [],
            "complexity" : [],
            "execStmt" : [],
            "maxNesting" : [],
        }
        metricsProcess = {
            "numOfCommits": [],
            "numOfCommittersUnique": [],
            "sumOfAdditionsLine": [],
            "sumOfDeletionsLine": [],
            "maxOfRatio_numOfChangesLineOfACommitter": [],
            "numOfCommittersUnfamiliar": [],
            "complexityHistory": [],
            "numOfCommitsNeighbor": [],
            "numOfCommittersUniqueNeighbor": [],
            "complexityHistoryNeighbor": [],
            "maxOfRatio_numOfChangesLineOfACommitter": [],
            "geometricmean_sumOfChangesLineByTheCommitter": [],

            "numOfCommitsRefactoring": [],
            "numOfCommitsFixingBugs": [],
            "maxOfAdditionsLine": [],
            "avgOfAdditionsLine": [],
            "maxOfDeletionsLine": [],
            "avgOfDeletionsLine": [],
            "sumOfChurnLine": [],
            "maxOfChurnLine": [],
            "avgOfChurnLine": [],
            "maxOfModulesCommittedSimultaneously": [],
            "avgOfModulesCommittedSimultaneously": [],
            "periodExisting": [],
            "periodExistingWeighted": [],

            "sumOfChangesDeclarationItself": [],
            "sumOfChangesStatement": [],
            "sumOfChangesCondition": [],
            "sumOfChangesStatementElse": []
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
            "fanIn" : [],
            "fanOut" : [],
            "parameters" : [],
            "localVar" : [],
            "commentRatio" : [],
            "countPath" : [],
            "complexity" : [],
            "execStmt" : [],
            "maxNesting" : [],
        }
        map2StandardizeMetricsProcess = {
            "numOfCommits": [],
            "numOfCommittersUnique": [],
            "sumOfAdditionsLine": [],
            "sumOfDeletionsLine": [],
            "maxOfRatio_numOfChangesLineOfACommitter": [],
            "numOfCommittersUnfamiliar": [],
            "complexityHistory": [],
            "numOfCommitsNeighbor": [],
            "numOfCommittersUniqueNeighbor": [],
            "complexityHistoryNeighbor": [],
            "maxOfRatio_numOfChangesLineOfACommitter": [],
            "geometricmean_sumOfChangesLineByTheCommitter": [],

            "numOfCommitsRefactoring": [],
            "numOfCommitsFixingBugs": [],
            "maxOfAdditionsLine": [],
            "avgOfAdditionsLine": [],
            "maxOfDeletionsLine": [],
            "avgOfDeletionsLine": [],
            "sumOfChurnLine": [],
            "maxOfChurnLine": [],
            "avgOfChurnLine": [],
            "maxOfModulesCommittedSimultaneously": [],
            "avgOfModulesCommittedSimultaneously": [],
            "periodExisting": [],
            "periodExistingWeighted": [],

            "sumOfChangesDeclarationItself": [],
            "sumOfChangesStatement": [],
            "sumOfChangesCondition": [],
            "sumOfChangesStatementElse": []
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
        if(config.checkCodeMetricsExists()):
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    for item in metricsCode:
                        metricsCode[item].append(sampleJson[item])
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