from src.config.cfg import cfg
from src.data.Dataset import Dataset

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
        self.selectedInputData = cfg.typesInput

    def setPathsSample(self, pathsSample, isForTest=False):
        if(isForTest):
            for path in pathsSample:
                if(os.path.isdir(path)):
                    pathsSearched = glob.glob(path+"/**/*.json", recursive=True)
                    self.pathsSample4Test.extend(pathsSearched)
                else:
                    self.pathsSample4Test.append(path)
        else:
            for path in pathsSample:
                if(os.path.isdir(path)):
                    pathsSearched = glob.glob(path+"/**/*.json", recursive=True)
                    self.pathsSample4Train.extend(pathsSearched)
                else:
                    self.pathsSample4Train.append(path)

    def loadSamples(self):
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
                sample["y"] = sampleJson["isBuggy"]
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
                if(cfg.checkASTExists()):
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
                if(cfg.checkASTSeqExists()):
                    def onehotnizeAST(ast):
                        astOnehotnized = []
                        for node in ast:
                            nodeOnehotnized = onehotnizeASTNode(node["numType"])
                            astOnehotnized.append(nodeOnehotnized)
                        return astOnehotnized
                    nodes = flattenTree(sampleJson["ast"])
                    sample["x"]["astseq"] = onehotnizeAST(nodes)
                if(cfg.checkCodeMetricsExists()):
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
                if(cfg.checkCommitGraphExists()):
                    numOfCommits = len(sampleJson["commitGraph"])
                    sample["x"]["commitgraph"]["nodes"] = [[None] for i in range(numOfCommits)]
                    for i in range(numOfCommits):
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
                if(cfg.checkCommitSeqExists()):
                    commits = []
                    for commit in sampleJson["commitGraph"]["modifications"].values():
                        if(not commit["isMerge"]):
                            commits.append(commit)
                    commits = sorted(commits, key=lambda x: x['date'])
                    numOfCommits = len(commits)
                    sample["x"]["commitseq"] = [None] * numOfCommits
                    if(0<numOfCommits):
                        for i in range(numOfCommits):
                            sample["x"]["commitseq"][i] = [
                                (float(commits[i]["stmtAdded"])-map2StandardizeMetricsCommit["stmtAdded"][0])/map2StandardizeMetricsCommit["stmtAdded"][1],
                                (float(commits[i]["stmtDeleted"])-map2StandardizeMetricsCommit["stmtDeleted"][0])/map2StandardizeMetricsCommit["stmtDeleted"][1],
                                (float(commits[i]["churn"])-map2StandardizeMetricsCommit["churn"][0])/map2StandardizeMetricsCommit["churn"][1],
                                (float(commits[i]["decl"])-map2StandardizeMetricsCommit["decl"][0])/map2StandardizeMetricsCommit["decl"][1],
                                (float(commits[i]["cond"])-map2StandardizeMetricsCommit["cond"][0])/map2StandardizeMetricsCommit["cond"][1],
                                (float(commits[i]["elseAdded"])-map2StandardizeMetricsCommit["elseAdded"][0])/map2StandardizeMetricsCommit["elseAdded"][1],
                                (float(commits[i]["elseDeleted"])-map2StandardizeMetricsCommit["elseDeleted"][0])/map2StandardizeMetricsCommit["elseDeleted"][1]
                            ]
                    else:
                        sample["x"]["commitseq"] = [[0, 0, 0, 0, 0, 0, 0]]
                if(cfg.checkProcessMetricsExists()):
                    sample["x"]["processmetrics"].extend(
                        [
                            (float(sampleJson["commitGraph"]["moduleHistories"])-map2StandardizeMetricsProcess["moduleHistories"][0]) / map2StandardizeMetricsProcess["moduleHistories"][1],
                            (float(sampleJson["commitGraph"]["authors"])-map2StandardizeMetricsProcess["authors"][0]) / map2StandardizeMetricsProcess["authors"][1],
                            (float(sampleJson["commitGraph"]["sumStmtAdded"])-map2StandardizeMetricsProcess["sumStmtAdded"][0]) / map2StandardizeMetricsProcess["sumStmtAdded"][1],
                            (float(sampleJson["commitGraph"]["maxStmtAdded"])-map2StandardizeMetricsProcess["maxStmtAdded"][0]) / map2StandardizeMetricsProcess["maxStmtAdded"][1],
                            (float(sampleJson["commitGraph"]["avgStmtAdded"])-map2StandardizeMetricsProcess["avgStmtAdded"][0]) / map2StandardizeMetricsProcess["avgStmtAdded"][1],
                            (float(sampleJson["commitGraph"]["sumStmtDeleted"])-map2StandardizeMetricsProcess["sumStmtDeleted"][0]) / map2StandardizeMetricsProcess["sumStmtDeleted"][1],
                            (float(sampleJson["commitGraph"]["maxStmtDeleted"])-map2StandardizeMetricsProcess["maxStmtDeleted"][0]) / map2StandardizeMetricsProcess["maxStmtDeleted"][1],
                            (float(sampleJson["commitGraph"]["avgStmtDeleted"])-map2StandardizeMetricsProcess["avgStmtDeleted"][0]) / map2StandardizeMetricsProcess["avgStmtDeleted"][1],
                            (float(sampleJson["commitGraph"]["sumChurn"])-map2StandardizeMetricsProcess["sumChurn"][0]) / map2StandardizeMetricsProcess["sumChurn"][1],
                            (float(sampleJson["commitGraph"]["maxChurn"])-map2StandardizeMetricsProcess["maxChurn"][0]) / map2StandardizeMetricsProcess["maxChurn"][1],
                            (float(sampleJson["commitGraph"]["avgChurn"])-map2StandardizeMetricsProcess["avgChurn"][0]) / map2StandardizeMetricsProcess["avgChurn"][1],
                            (float(sampleJson["commitGraph"]["sumDecl"])-map2StandardizeMetricsProcess["sumDecl"][0]) / map2StandardizeMetricsProcess["sumDecl"][1],
                            (float(sampleJson["commitGraph"]["sumCond"])-map2StandardizeMetricsProcess["sumCond"][0]) / map2StandardizeMetricsProcess["sumCond"][1],
                            (float(sampleJson["commitGraph"]["sumElseAdded"])-map2StandardizeMetricsProcess["sumElseAdded"][0]) / map2StandardizeMetricsProcess["sumElseAdded"][1],
                            (float(sampleJson["commitGraph"]["sumElseDeleted"])-map2StandardizeMetricsProcess["sumElseDeleted"][0]) / map2StandardizeMetricsProcess["sumElseDeleted"][1]
                        ]
                    )
            listSamples.append(sample)
        metricsCommit = {
            "stmtAdded": [],
            "stmtDeleted": [],
            "churn": [],
            "decl": [],
            "cond": [],
            "elseAdded": [],
            "elseDeleted": []
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
            "moduleHistories" : [],
            "authors" : [],
            "sumStmtAdded" : [],
            "maxStmtAdded" : [],
            "avgStmtAdded" : [],
            "sumStmtDeleted" : [],
            "maxStmtDeleted" : [],
            "avgStmtDeleted" : [],
            "sumChurn" : [],
            "maxChurn" : [],
            "avgChurn" : [],
            "sumDecl" : [],
            "sumCond" : [],
            "sumElseAdded" : [],
            "sumElseDeleted" : [],
        }
        map2StandardizeMetricsCommit = {
            "stmtAdded": [],
            "stmtDeleted": [],
            "churn": [],
            "decl": [],
            "cond": [],
            "elseAdded": [],
            "elseDeleted": []
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
            "moduleHistories" : [],
            "authors" : [],
            "sumStmtAdded" : [],
            "maxStmtAdded" : [],
            "avgStmtAdded" : [],
            "sumStmtDeleted" : [],
            "maxStmtDeleted" : [],
            "avgStmtDeleted" : [],
            "sumChurn" : [],
            "maxChurn" : [],
            "avgChurn" : [],
            "sumDecl" : [],
            "sumCond" : [],
            "sumElseAdded" : [],
            "sumElseDeleted" : [],
        }
        if(cfg.checkCommitSeqExists()):
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    for commit in sampleJson["commitGraph"]["modifications"].values():
                        for item in metricsCommit:
                            metricsCommit[item].append(commit[item])
            for item in map2StandardizeMetricsCommit:
                map2StandardizeMetricsCommit[item] = [np.array(metricsCommit[item]).mean(), np.std(metricsCommit[item])]
        if(cfg.checkCodeMetricsExists()):
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    for item in metricsCode:
                        metricsCode[item].append(sampleJson[item])
            for item in map2StandardizeMetricsCode:
                map2StandardizeMetricsCode[item] = [np.array(metricsCode[item]).mean(), np.std(metricsCode[item])]
        if(cfg.checkProcessMetricsExists()):
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    for item in metricsProcess:
                        metricsProcess[item].append(sampleJson["commitGraph"][item])
            for item in map2StandardizeMetricsProcess:
                map2StandardizeMetricsProcess[item] = [np.array(metricsProcess[item]).mean(), np.std(metricsProcess[item])]
        for pathSample4Train in self.pathsSample4Train:
            loadSample(pathSample4Train, self.samples4Train, map2StandardizeMetricsCommit, map2StandardizeMetricsCode, map2StandardizeMetricsProcess)
        for pathSample4Test in self.pathsSample4Test:
            loadSample(pathSample4Test, self.samples4Test, map2StandardizeMetricsCommit, map2StandardizeMetricsCode, map2StandardizeMetricsProcess)

    def generateDatasetsTrainValid(self, isCrossValidation = False, numOfSplit = 5):
        samplesBuggy    = []
        samplesNotBuggy = []
        for data in self.samples4Train:
            if(data["y"]==1):
                samplesBuggy.append(data)
            elif(int(data["y"])==0):
                samplesNotBuggy.append(data)
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
            dataset_train_valid["train"] = Dataset(dataset4Train)
            dataset_train_valid["valid"] = Dataset(dataset4Valid)
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
        samplesBuggy = random.choices(samplesBuggy, k=len(samplesNotBuggy))
        dataset4Train.extend(samplesBuggy)
        dataset4Train.extend(samplesNotBuggy)
        random.shuffle(dataset4Train)#最初に1, 次に0ばっかり並んでしまうのを防ぐ。
        dataset4Test = self.samples4Test
        dataset_Train_Test["train"] = Dataset(dataset4Train)
        dataset_Train_Test["test"] = Dataset(dataset4Test)
        self.datasets_Train_Test = dataset_Train_Test

    def showSummary(self):
        print(" len(samples4Train): " + str(len(self.samples4Train)))
        print(" len(samples4Test): " + str(len(self.samples4Test)))