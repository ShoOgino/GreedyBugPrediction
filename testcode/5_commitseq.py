import sys, os
sys.path += [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/sourcecode"]
print(sys.path)
from src.manager import Maneger
from src.config.cfg import cfg
import datetime


# cfg(タスクの設定)を更新
cfg.clear()
cfg.pathConfigFile              = os.path.abspath(__file__)
cfg.project                     = "example"
cfg.purpose                     = [cfg.Purpose.searchHyperParameter, cfg.Purpose.searchParameter, cfg.Purpose.test]
cfg.typesInput                  = [cfg.TypeInput.commitseq]
cfg.pathsSampleTrain            = [os.path.abspath("../dataset/{}/4train".format(cfg.project))]
cfg.pathsSampleTest             = [os.path.abspath("../dataset/{}/4test".format(cfg.project))]
cfg.adoptingCrossValidation     = True
cfg.splitSize4Validation        = 5
cfg.epochs4EarlyStopping        = 50
cfg.period4HyperParameterSearch = 60*60*1 #seconds
cfg.id                          = os.path.splitext(os.path.basename(cfg.pathConfigFile))[0] + "_" + cfg.project + "_"
cfg.pathDirOutput               = os.path.dirname(os.path.dirname(cfg.pathConfigFile)) + "/results/" + cfg.id + "_"+str(datetime.datetime.today().strftime("%Y%m%d_%H%M%S"))
#cfg.pathLogSearchHyperParameter = r""

# 実験を実行
maneger = Maneger()
maneger.run()