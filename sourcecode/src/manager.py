from src.data.DataManeger import DataManeger
from src.model.Modeler import Modeler
from src.config.cfg import cfg

import os
import shutil

class Maneger:
    def __init__(self):
        pass

    def run(self):
        os.makedirs(cfg.pathDirOutput, exist_ok=True)
        shutil.copy(cfg.pathConfigFile, cfg.pathDirOutput) #実験結果フォルダへ実行環境情報を保存
        if(cfg.pathLogSearchHyperParameter!=""): #引き継ぐoptunaDBをコピー
            shutil.copy(cfg.pathLogSearchHyperParameter, cfg.pathDirOutput)

        # データ職人生成
        dataManeger = DataManeger()
        dataManeger.setPathsSample(cfg.pathsSampleTrain, False)
        dataManeger.setPathsSample(cfg.pathsSampleTest, True)
        dataManeger.loadSamples()

        # モデル職人生成
        modeler = Modeler()

        # データ職人・モデル職人にタスクを移譲
        if(cfg.checkPurposeContainsSearchHyperParameter()):
            print("-----searchHyperParameter-----")
            dataManeger.generateDatasetsTrainValid(isCrossValidation = cfg.isCrossValidation, numOfSplit = cfg.splitSize4CrossValidation)
            modeler.searchHyperParameter(dataManeger.datasets_Train_Valid)
        if(cfg.checkPurposeContainsSearchParameter()):
            print("-----searchParameter----------")
            dataManeger.generateDatasetsTrainTest()
            modeler.searchParameter(dataManeger.datasets_Train_Test)
        if(cfg.checkPurposeContainsTest()):
            print("-----test---------------------")
            dataManeger.generateDatasetsTrainTest()
            modeler.test(dataManeger.datasets_Train_Test)