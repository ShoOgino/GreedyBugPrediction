from src.config.config import config
import os
import datetime

if __name__ == '__main__':
    # config(タスクの設定)を更新
    config.pathConfigFile              = os.path.abspath(__file__)
    config.project                     = "egit"
    config.release                     = 2
    config.purpose                     = [
        config.Purpose.searchHyperParameter,
        config.Purpose.buildModel,
        config.Purpose.testModel
    ]
    config.typesInput                  = [
        config.TypeInput.processmetrics
    ]
    config.pathsDirSampleTrain         = [
        "C:/Users/login/data/workspace/MLTool/datasets/{}/outputFile/R{}_r_train".format(config.project, config.release)
    ]
    config.pathsDirSampleTest          = [
        "C:/Users/login/data/workspace/MLTool/datasets/{}/outputFile/R{}_r_test".format(config.project, config.release)
    ]
    config.isCrossValidation           = True
    config.splitSize4CrossValidation   = 5
    config.epochs4EarlyStopping        = 100
    config.period4HyperParameterSearch = 60 #seconds
    config.id                          = os.path.splitext(os.path.basename(config.pathConfigFile))[0] + "_" + config.project + "_" + str(config.release)
    config.pathDirOutput               = os.path.dirname(os.path.dirname(config.pathConfigFile)) + "/results/" + config.id + "_"+str(datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))
    config.pathLog = config.pathDirOutput+"/log.txt"

    # 実験を実行
    from src.manager import Maneger
    maneger = Maneger()
    maneger.run()