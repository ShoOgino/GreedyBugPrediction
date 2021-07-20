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
        def loadSample(pathSample, listSamples, module2Num, committer2Num, tableInterval, tableChurn, map2StandardizeMetricsCode, map2StandardizeMetricsProcess):
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
                            (float(sampleJson["fanIn"])-map2StandardizeMetricsCode["fanIn"][0]) / map2StandardizeMetricsCode["fanIn"][1],
                            (float(sampleJson["fanOut"])-map2StandardizeMetricsCode["fanOut"][0]) / map2StandardizeMetricsCode["fanOut"][1],
                            (float(sampleJson["parameters"])-map2StandardizeMetricsCode["parameters"][0]) / map2StandardizeMetricsCode["parameters"][1],
                            (float(sampleJson["localVar"])-map2StandardizeMetricsCode["localVar"][0]) / map2StandardizeMetricsCode["localVar"][1],
                            (float(sampleJson["commentRatio"])-map2StandardizeMetricsCode["commentRatio"][0]) / map2StandardizeMetricsCode["commentRatio"][1],
                            (float(sampleJson["countPath"])-map2StandardizeMetricsCode["countPath"][0]) / map2StandardizeMetricsCode["countPath"][1],
                            (float(sampleJson["complexity"])-map2StandardizeMetricsCode["complexity"][0]) / map2StandardizeMetricsCode["complexity"][1],
                            (float(sampleJson["execStmt"])-map2StandardizeMetricsCode["execStmt"][0]) / map2StandardizeMetricsCode["execStmt"][1],
                            (float(sampleJson["maxNesting"])-map2StandardizeMetricsCode["maxNesting"][0]) / map2StandardizeMetricsCode["maxNesting"][1],
                            (float(sampleJson["loc"])-map2StandardizeMetricsCode["loc"][0]) / map2StandardizeMetricsCode["loc"][1],
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
                    numOfCommits = len(sampleJson["commitGraph"])
                    sample["x"]["commitseq"] = [None] * numOfCommits
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
                        sample["x"]["commitseq"][node["num"]] = vector[0]+vector[1]+vector[2]+vector[3]+vector[4][0]+vector[4][1]+vector[4][2]+vector[5]
                if(cfg.checkProcessMetricsExists()):
                    sample["x"]["processmetrics"].extend(
                        [
                            (float(sampleJson["moduleHistories"])-map2StandardizeMetricsProcess["moduleHistories"][0]) / map2StandardizeMetricsProcess["moduleHistories"][1],
                            (float(sampleJson["authors"])-map2StandardizeMetricsProcess["authors"][0]) / map2StandardizeMetricsProcess["authors"][1],
                            (float(sampleJson["stmtAdded"])-map2StandardizeMetricsProcess["stmtAdded"][0]) / map2StandardizeMetricsProcess["stmtAdded"][1],
                            (float(sampleJson["maxStmtAdded"])-map2StandardizeMetricsProcess["maxStmtAdded"][0]) / map2StandardizeMetricsProcess["maxStmtAdded"][1],
                            (float(sampleJson["avgStmtAdded"])-map2StandardizeMetricsProcess["avgStmtAdded"][0]) / map2StandardizeMetricsProcess["avgStmtAdded"][1],
                            (float(sampleJson["stmtDeleted"])-map2StandardizeMetricsProcess["stmtDeleted"][0]) / map2StandardizeMetricsProcess["stmtDeleted"][1],
                            (float(sampleJson["maxStmtDeleted"])-map2StandardizeMetricsProcess["maxStmtDeleted"][0]) / map2StandardizeMetricsProcess["maxStmtDeleted"][1],
                            (float(sampleJson["avgStmtDeleted"])-map2StandardizeMetricsProcess["avgStmtDeleted"][0]) / map2StandardizeMetricsProcess["avgStmtDeleted"][1],
                            (float(sampleJson["churn"])-map2StandardizeMetricsProcess["churn"][0]) / map2StandardizeMetricsProcess["churn"][1],
                            (float(sampleJson["maxChurn"])-map2StandardizeMetricsProcess["maxChurn"][0]) / map2StandardizeMetricsProcess["maxChurn"][1],
                            (float(sampleJson["avgChurn"])-map2StandardizeMetricsProcess["avgChurn"][0]) / map2StandardizeMetricsProcess["avgChurn"][1],
                            (float(sampleJson["decl"])-map2StandardizeMetricsProcess["decl"][0]) / map2StandardizeMetricsProcess["decl"][1],
                            (float(sampleJson["cond"])-map2StandardizeMetricsProcess["cond"][0]) / map2StandardizeMetricsProcess["cond"][1],
                            (float(sampleJson["elseAdded"])-map2StandardizeMetricsProcess["elseAdded"][0]) / map2StandardizeMetricsProcess["elseAdded"][1],
                            (float(sampleJson["elseDeleted"])-map2StandardizeMetricsProcess["elseDeleted"][0]) / map2StandardizeMetricsProcess["elseDeleted"][1],
                            (float(sampleJson["addLOC"])-map2StandardizeMetricsProcess["addLOC"][0]) / map2StandardizeMetricsProcess["addLOC"][1],
                            (float(sampleJson["delLOC"])-map2StandardizeMetricsProcess["delLOC"][0]) / map2StandardizeMetricsProcess["delLOC"][1],
                            (float(sampleJson["devMinor"])-map2StandardizeMetricsProcess["devMinor"][0]) / map2StandardizeMetricsProcess["devMinor"][1],
                            (float(sampleJson["devMajor"])-map2StandardizeMetricsProcess["devMajor"][0]) / map2StandardizeMetricsProcess["devMajor"][1],
                            (float(sampleJson["ownership"])-map2StandardizeMetricsProcess["ownership"][0]) / map2StandardizeMetricsProcess["ownership"][1],
                            (float(sampleJson["fixChgNum"])-map2StandardizeMetricsProcess["fixChgNum"][0]) / map2StandardizeMetricsProcess["fixChgNum"][1],
                            (float(sampleJson["pastBugNum"])-map2StandardizeMetricsProcess["pastBugNum"][0]) / map2StandardizeMetricsProcess["pastBugNum"][1],
                            (float(sampleJson["bugIntroNum"])-map2StandardizeMetricsProcess["bugIntroNum"][0]) / map2StandardizeMetricsProcess["bugIntroNum"][1],
                            (float(sampleJson["logCoupNum"])-map2StandardizeMetricsProcess["logCoupNum"][0]) / map2StandardizeMetricsProcess["logCoupNum"][1],
                            (float(sampleJson["period"])-map2StandardizeMetricsProcess["period"][0]) / map2StandardizeMetricsProcess["period"][1],
                            (float(sampleJson["avgInterval"])-map2StandardizeMetricsProcess["avgInterval"][0]) / map2StandardizeMetricsProcess["avgInterval"][1],
                            (float(sampleJson["maxInterval"])-map2StandardizeMetricsProcess["maxInterval"][0]) / map2StandardizeMetricsProcess["maxInterval"][1],
                            (float(sampleJson["minInterval"])-map2StandardizeMetricsProcess["minInterval"][0]) / map2StandardizeMetricsProcess["minInterval"][1],
                        ]
                    )
            listSamples.append(sample)
            #name, ext = os.path.splitext(pathSample)
            #pathVector = name+"_vector"
            #with open(pathVector, "w") as f:
            #    json.dump(sample, f)
        # 学習時のモジュールIDと総数、コミッターIDと総数、intervalの離散化範囲、churnの離散化範囲、各種メトリクスの標準化のための値の算出
        module2Num = {"other": 0}#id==0: その他
        committer2Num = {"other": 0}#id==0: その他
        tableInterval = [1000000]*50
        tableChurn = [[1000000]*50, [1000000]*50, [1000000]*50]
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
            "loc":[]
        }
        metricsProcess = {
            "moduleHistories" : [],
            "authors" : [],
            "stmtAdded" : [],
            "maxStmtAdded" : [],
            "avgStmtAdded" : [],
            "stmtDeleted" : [],
            "maxStmtDeleted" : [],
            "avgStmtDeleted" : [],
            "churn" : [],
            "maxChurn" : [],
            "avgChurn" : [],
            "decl" : [],
            "cond" : [],
            "elseAdded" : [],
            "elseDeleted" : [],
            "addLOC" : [],
            "delLOC" : [],
            "devMinor" : [],
            "devMajor" : [],
            "ownership" : [],
            "fixChgNum" : [],
            "pastBugNum" : [],
            "bugIntroNum" : [],
            "logCoupNum" : [],
            "period" : [],
            "avgInterval" : [],
            "maxInterval" : [],
            "minInterval" : []
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
            "loc":[]
        }
        map2StandardizeMetricsProcess = {
            "moduleHistories" : [],
            "authors" : [],
            "stmtAdded" : [],
            "maxStmtAdded" : [],
            "avgStmtAdded" : [],
            "stmtDeleted" : [],
            "maxStmtDeleted" : [],
            "avgStmtDeleted" : [],
            "churn" : [],
            "maxChurn" : [],
            "avgChurn" : [],
            "decl" : [],
            "cond" : [],
            "elseAdded" : [],
            "elseDeleted" : [],
            "addLOC" : [],
            "delLOC" : [],
            "devMinor" : [],
            "devMajor" : [],
            "ownership" : [],
            "fixChgNum" : [],
            "pastBugNum" : [],
            "bugIntroNum" : [],
            "logCoupNum" : [],
            "period" : [],
            "avgInterval" : [],
            "maxInterval" : [],
            "minInterval" : [],
        }
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
                        metricsProcess[item].append(sampleJson[item])
            for item in map2StandardizeMetricsProcess:
                map2StandardizeMetricsProcess[item] = [np.array(metricsProcess[item]).mean(), np.std(metricsProcess[item])]
        if(cfg.checkCommitGraphExists() or cfg.checkCommitSeqExists()):
            numOfAllNodes = 0
            intervals = [0]*1000
            churns = [[0]*1000, [0]*1000, [0]*1000]
            for pathSample4Train in self.pathsSample4Train:
                with open(pathSample4Train, encoding="utf-8") as fSample4Train:
                    sampleJson = json.load(fSample4Train)
                    numOfNodes = len(sampleJson["commitGraph"])
                    # 学習時のモジュールIDについて
                    if(not sampleJson["path"] in module2Num):
                        module2Num[sampleJson["path"]] = len(module2Num)
                    for i in range(numOfNodes):
                        node = sampleJson["commitGraph"][i]
                        # 学習時のコミッターIDについて
                        if(not node["author"] in committer2Num):
                            committer2Num[node["author"]] = len(committer2Num)
                        if(node["num"]==0):
                            continue
                        else:
                            numOfAllNodes += 1
                            if(1000<=node["interval"]):
                                intervals[999] += 1
                            else:
                                intervals[node["interval"]] += 1
                            if(1000<=node["churn"][0]):
                                churns[0][999] += 1
                            else:
                                churns[0][node["churn"][0]] += 1
                            if(1000<=node["churn"][1]):
                                churns[1][999] += 1
                            else:
                                churns[1][node["churn"][1]] += 1
                            if(1000<=node["churn"][2]):
                                churns[2][999] += 1
                            else:
                                churns[2][node["churn"][2]] += 1
            numOfNodesPre = 0
            numOfNodesNow = 0
            index = 0
            for i in range(len(intervals)):
                numOfNodesNow += intervals[i]
                if( (numOfAllNodes/50)*index < numOfNodesNow-numOfNodesPre):
                    tableInterval[index] = i
                    numOfNodesPre = numOfNodesNow
                    index+=1
            numOfNodesPre = 0
            numOfNodesNow = 0
            index = 0
            for i in range(len(churns[0])):
                numOfNodesNow += churns[0][i]
                if( (numOfAllNodes/50)*index < numOfNodesNow-numOfNodesPre):
                    tableChurn[0][index] = i
                    numOfNodesPre = numOfNodesNow
                    index+=1
            numOfNodesPre = 0
            numOfNodesNow = 0
            index = 0
            for i in range(len(churns[1])):
                numOfNodesNow += churns[1][i]
                if( (numOfAllNodes/50)*index < numOfNodesNow-numOfNodesPre):
                    tableChurn[1][index] = i
                    numOfNodesPre = numOfNodesNow
                    index+=1
            numOfNodesPre = 0
            numOfNodesNow = 0
            index = 0
            for i in range(len(churns[2])):
                numOfNodesNow += churns[2][i]
                if( (numOfAllNodes/50)*index < numOfNodesNow-numOfNodesPre):
                    tableChurn[2][index] = i
                    numOfNodesPre = numOfNodesNow
                    index+=1
        for pathSample4Train in self.pathsSample4Train:
            loadSample(pathSample4Train, self.samples4Train, module2Num, committer2Num, tableInterval, tableChurn, map2StandardizeMetricsCode, map2StandardizeMetricsProcess)
        for pathSample4Test in self.pathsSample4Test:
            loadSample(pathSample4Test, self.samples4Test, module2Num, committer2Num, tableInterval, tableChurn, map2StandardizeMetricsCode, map2StandardizeMetricsProcess)

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