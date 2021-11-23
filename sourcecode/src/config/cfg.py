from enum import Enum
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing

class cfg:
    class Purpose(Enum):
        searchHyperParameter = 1
        searchParameter = 2
        test = 3
    class TypeInput(Enum):
        mnist4test = 0
        ast = 1
        astseq = 2
        codemetrics = 3
        commitgraph = 4
        commitseq = 5
        processmetrics = 6

    def clear():
        cfg.id = None
        cfg.pathConfigFile = None
        cfg.purpose = None
        cfg.typesInput = None
        cfg.pathsSampleTrain = None # list
        cfg.pathsSampleTest = None # list
        cfg.isCrossValidation = False
        cfg.splitSize4CrossValidation = 5
        cfg.epochsEarlyStopping = 10
        cfg.trials4HyperParameterSearch = None
        cfg.period4HyperParameterSearch = None
        cfg.device = "cuda:0"
        cfg.pathLogSearchHyperParameter = ""
        cfg.pathHyperParameters = ""
        cfg.pathParameters = ""
        cfg.pathDirOutput = None

    def checkPurposeContainsSearchHyperParameter():
        return cfg.Purpose.searchHyperParameter in cfg.purpose
    def checkPurposeContainsSearchParameter():
        return cfg.Purpose.searchParameter in cfg.purpose
    def checkPurposeContainsTest():
        return cfg.Purpose.test in cfg.purpose

    def checkMnist4TestExists():
        return cfg.TypeInput.mnist4test in cfg.typesInput
    def checkASTExists():
        return cfg.TypeInput.ast in cfg.typesInput
    def checkASTSeqExists():
        return cfg.TypeInput.astseq in cfg.typesInput
    def checkCodeMetricsExists():
        return cfg.TypeInput.codemetrics in cfg.typesInput
    def checkCommitGraphExists():
        return cfg.TypeInput.commitgraph in cfg.typesInput
    def checkCommitSeqExists():
        return cfg.TypeInput.commitseq in cfg.typesInput
    def checkProcessMetricsExists():
        return cfg.TypeInput.processmetrics in cfg.typesInput