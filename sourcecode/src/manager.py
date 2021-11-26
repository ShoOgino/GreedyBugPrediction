from src.data.DataManeger import DataManeger
from src.model.Modeler import Modeler
from src.config.config import config
from src.log.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.pathLog)

import os
import shutil

class Maneger:
    def __init__(self):
        pass

    def run(self):
        os.makedirs(config.pathDirOutput, exist_ok=True)
        # 実験結果フォルダへ実行コードを保存
        shutil.copy(config.pathConfigFile, config.pathDirOutput)
        # 引き継ぐoptunaDBをコピー
        if(config.pathDatabaseOptuna!=None):
            shutil.copy(config.pathDatabaseOptuna, config.pathDirOutput)

        # データ職人生成
        dataManeger = DataManeger()
        dataManeger.setPathsSample(config.pathsSampleTrain, False)
        dataManeger.setPathsSample(config.pathsSampleTest, True)
        dataManeger.loadSamples()

        # モデル職人生成
        modeler = Modeler()

        # データ職人・モデル職人にタスクを移譲
        if(config.checkPurposeContainsSearchHyperParameter()):
            dataManeger.generateDatasetsTrainValid(isCrossValidation = config.isCrossValidation, numOfSplit = config.splitSize4CrossValidation)
            modeler.searchHyperParameter(dataManeger.datasets_Train_Valid)
        if(config.checkPurposeContainsBuildModel()):
            dataManeger.generateDatasetsTrainTest()
            modeler.buildModel(dataManeger.datasets_Train_Test)
        if(config.checkPurposeContainsTest()):
            dataManeger.generateDatasetsTrainTest()
            modeler.test(dataManeger.datasets_Train_Test["test"])