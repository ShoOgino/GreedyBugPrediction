from src.manager import Maneger
from src.config.cfg import cfg

import os
import glob
import shutil
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing
import datetime

# cfg(タスクの設定)を更新
cfg.clear()
cfg.pathConfigFile              = os.path.abspath(__file__)
cfg.project                     = "cassandra"
cfg.release                     = 2
#cfg.purpose                     = [cfg.Purpose.searchHyperParameter, cfg.Purpose.searchParameter, cfg.Purpose.test]
cfg.purpose                     = [cfg.Purpose.searchParameter, cfg.Purpose.test]
cfg.typesInput                  = [cfg.TypeInput.codemetrics, cfg.TypeInput.processmetrics]
cfg.pathsSampleTrain            = ["C:/Users/login/data/workspace/MLTool/datasets/{}/output/R{}_r_train".format(cfg.project, cfg.release)]
cfg.pathsSampleTest             = ["C:/Users/login/data/workspace/MLTool/datasets/{}/output/R{}_r_test".format(cfg.project, cfg.release)]
cfg.adoptingCrossValidation     = False
cfg.splitSize4Validation        = 5
cfg.epochs4EarlyStopping        = 10
cfg.period4HyperParameterSearch = 60*60*24
cfg.id                          = os.path.splitext(os.path.basename(cfg.pathConfigFile))[0] + "_" + cfg.project + "_" + str(cfg.release)
cfg.pathDirOutput               = os.path.dirname(os.path.dirname(cfg.pathConfigFile)) + "/results/" + cfg.id + "_"+str(datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))

# 実験を実行
maneger = Maneger()
maneger.run()