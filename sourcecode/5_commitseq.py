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
        config.TypeInput.commitseq
    ]
    config.pathsDirSampleTrain         = [
        "../datasets/{}/R{}_r_train".format(config.project, config.release)
    ]
    config.pathsDirSampleTest          = [
        "../datasets/{}/R{}_r_test".format(config.project, config.release)
    ]
    config.algorithm                   = "DNN"
    config.isCrossValidation           = False
    config.splitSize4CrossValidation   = 5
    config.epochs4EarlyStopping        = 10
    config.period4HyperParameterSearch = 60 #seconds
    config.id                          = os.path.splitext(os.path.basename(config.pathConfigFile))[0] + "_" + config.project + "_" + str(config.release)
    config.pathDirOutput               = os.path.dirname(os.path.dirname(config.pathConfigFile)) + "/results/" + config.id + "_"+str(datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))

    # 実験を実行
    from src.manager import Maneger
    maneger = Maneger()
    maneger.run()