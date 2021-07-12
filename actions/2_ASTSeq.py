from src.manager import Maneger
from src.config.cfg import cfg

import os
import glob
import shutil
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing

# cfg(タスクの設定)を更新
cfg.clear()
cfg.project                     = "egit"
cfg.purpose                     = ["searchHyperParameter", "searchParameter", "test"]
#cfg.purpose                     = ["searchHyperParameter", "searchParameter", "test"]
cfg.typesInput                  = ["codemetrics", "processmetrics"]#["ast", "astseq", "codemetrics", "commitgraph", "commitseq", "processmetrics"]
#cfg.typesInput                  = ["astseq", "commitseq"]#["ast", "astseq", "codemetrics", "commitgraph", "commitseq", "processmetrics"]
cfg.pathsSampleTrain            = [r"C:\Users\login\data\workspace\MLTool\datasets\egit\output\R2_r_train"]
cfg.pathsSampleTest             = [r"C:\Users\login\data\workspace\MLTool\datasets\egit\output\R2_r_test"]
cfg.adoptingCrossValidation     = False
cfg.splitSize4Validation        = 5
cfg.epochs4EarlyStopping        = 10
cfg.period4HyperParameterSearch = 60*60*10
cfg.id                          = "ASTSeq_"+cfg.project
cfg.pathDirOutput               = os.path.dirname(os.path.dirname(__file__)) + "/results/" + cfg.id
cfg.pathHyperParameters         = ""#cfg.pathDirOutput + "/hyperparameter.json"
cfg.pathParameters              = ""

# 実験を実行
maneger = Maneger()
maneger.run()