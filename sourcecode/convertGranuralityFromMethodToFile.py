import argparse
import glob
import json
import os
import re


def reformat(pathDataset):
    methods = {}
    files = {}
    #1. メソッドについてのデータファイルをすべて読み込み、クラスごとにグループ化する
    pathMethods = glob.glob(pathDataset+"/**/*.json", recursive=True)
    for pathMethod in pathMethods:
        with open(pathMethod, "r", encoding="utf-8") as f:
            d = json.load(f)
            idMethod = d["path"]
            idFile = re.findall(".+(?=#)", idMethod)[0]
            key2commitOnMethod = d["commitGraph"]["modifications"]
            commitOnMethods = []
            for key in key2commitOnMethod:
                commitOnMethods.append(key2commitOnMethod[key])
            if(idFile not in files):
                files[idFile] = {}
                files[idFile]["isBuggy"] =0
                files[idFile]["commitOnFiles"] = []
                files[idFile]["commitOnFiles"].extend(commitOnMethods)
            else:
                files[idFile]["commitOnFiles"].extend(commitOnMethods)
            files[idFile]["isBuggy"] += d["isBuggy"]
            if(0<files[idFile]["isBuggy"]):
                files[idFile]["isBuggy"] = 1


    #4. 同じidを持つ変更ベクトルを一つのベクトルにまとめ、新たな変更ベクトル列を作る
    for idFile in files.keys():
        commitOnFilesBefore = {}
        commitOnFilesAfter = []
        for commitOnMethod in files[idFile]["commitOnFiles"]:
            if(not commitOnMethod["idCommit"] in commitOnFilesBefore):
                commitOnFilesBefore[commitOnMethod["idCommit"]] = []
                commitOnFilesBefore[commitOnMethod["idCommit"]].append(commitOnMethod)
            else:
                commitOnFilesBefore[commitOnMethod["idCommit"]].append(commitOnMethod)
        for idCommit in commitOnFilesBefore:
            commitOnFile = {}
            commitOnFile["idCommit"] = idCommit
            commitOnFile["date"] = commitOnFilesBefore[idCommit][0]["date"]
            commitOnFile["isMerge"] = commitOnFilesBefore[idCommit][0]["isMerge"]
            commitOnFile["stmtAdded"] = 0
            commitOnFile["stmtDeleted"] = 0
            commitOnFile["churn"] = 0
            commitOnFile["decl"] = 0
            commitOnFile["cond"] = 0
            commitOnFile["elseAdded"] = 0
            commitOnFile["elseDeleted"] = 0
            for commitOnFileOnIdcommit in commitOnFilesBefore[idCommit]:
                commitOnFile["stmtAdded"]   += commitOnFileOnIdcommit["stmtAdded"]
                commitOnFile["stmtDeleted"] += commitOnFileOnIdcommit["stmtDeleted"]
                commitOnFile["churn"]       += commitOnFileOnIdcommit["churn"]       
                commitOnFile["decl"]        += commitOnFileOnIdcommit["decl"]        
                commitOnFile["cond"]        += commitOnFileOnIdcommit["cond"]        
                commitOnFile["elseAdded"]   += commitOnFileOnIdcommit["elseAdded"]   
                commitOnFile["elseDeleted"] += commitOnFileOnIdcommit["elseDeleted"] 
            commitOnFilesAfter.append(commitOnFile)
        files[idFile]["commitOnFiles"] = sorted(commitOnFilesAfter, key=lambda x:x["date"])#5. 新しい変更ベクトル列を時間順にソートする
    #6. 新しい変更ベクトル列からメトリクスを算出する。
    for idFile in files.keys():
        files[idFile]["moduleHistories"]=0
        files[idFile]["sumStmtAdded"]=0
        files[idFile]["maxStmtAdded"]=0
        files[idFile]["avgStmtAdded"]=0
        files[idFile]["sumStmtDeleted"]=0
        files[idFile]["maxStmtDeleted"]=0
        files[idFile]["avgStmtDeleted"]=0
        files[idFile]["sumChurn"]=0
        files[idFile]["maxChurn"]=0
        files[idFile]["avgChurn"]=0
        files[idFile]["sumDecl"]=0
        files[idFile]["sumCond"]=0
        files[idFile]["sumElseAdded"]=0
        files[idFile]["sumElseDeleted"]=0
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["moduleHistories"]+=1
        if(files[idFile]["moduleHistories"] ==0):
            continue
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["sumStmtAdded"]+=commit["stmtAdded"]
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            if(files[idFile]["maxStmtAdded"] < commit["stmtAdded"]):
                files[idFile]["maxStmtAdded"] = commit["stmtAdded"]
        files[idFile]["avgStmtAdded"]=files[idFile]["sumStmtAdded"]/files[idFile]["moduleHistories"]

        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["sumStmtDeleted"]+=commit["stmtDeleted"]
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            if(files[idFile]["maxStmtDeleted"] < commit["stmtDeleted"]):
                files[idFile]["maxStmtDeleted"] = commit["stmtDeleted"]
        files[idFile]["avgStmtDeleted"]=files[idFile]["sumStmtDeleted"]/files[idFile]["moduleHistories"]

        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["sumChurn"]+=commit["churn"]
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            if(files[idFile]["maxChurn"] < commit["churn"]):
                files[idFile]["maxChurn"] = commit["churn"]
        files[idFile]["avgChurn"]=files[idFile]["sumChurn"]/files[idFile]["moduleHistories"]
        
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["sumDecl"]+=commit["decl"]
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["sumCond"]+=commit["cond"]
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["sumElseAdded"]+=commit["elseAdded"]
        for commit in files[idFile]["commitOnFiles"]:
            if(commit["isMerge"] == True): continue
            files[idFile]["sumElseDeleted"]+=commit["elseDeleted"]

    #reformat
    for idFile in files.keys():
        temp = {}
        temp["path"] = idFile
        temp["isBuggy"] = files[idFile]["isBuggy"]
        temp["commitGraph"]=files[idFile]
        temp["commitGraph"]["modifications"] = {}
        for commitOnFile in files[idFile]["commitOnFiles"]:
            temp["commitGraph"]["modifications"][commitOnFile["idCommit"]] = commitOnFile
        files[idFile] = temp


    #7. ファイルについてのデータをファイルとして書き込む
    for idFile in files.keys():
        path = idFile+".json"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(idFile+".json", "w", encoding="utf-8") as f:
            d = json.dumps(files[idFile])
            f.write(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Convert a git log output to json.""")
    parser.add_argument('--pathDataset', type=str, default="", help="dest")
    args = parser.parse_args()
    reformat(args.pathDataset)
