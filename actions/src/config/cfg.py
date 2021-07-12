from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing

class cfg:
    def clear():
        cfg.id = None
        cfg.purpose = ["searchHyperParameter", "searchParameter", "test"]
        cfg.typesInput = None # "ast", "astseq", "codemetrics", "commitgraph", "commitseq", "processmetrics"
        cfg.pathsSampleTrain = None # list
        cfg.pathsSampleTest = None # list
        cfg.isCrossValidation = False
        cfg.splitSize4Validation = 5
        cfg.epochsEarlyStopping = 10
        cfg.trials4HyperParameterSearch = None
        cfg.period4HyperParameterSearch = None
        cfg.device = "cuda:0"
        cfg.pathHyperParameters = ""
        cfg.pathParameters = ""
        cfg.pathDirOutput = None

    def checkPurposeContainsSearchHyperParameter():
        return "searchHyperParameter" in cfg.purpose
    def checkPurposeContainsSearchParameter():
        return "searchParameter" in cfg.purpose
    def checkPurposeContainsTest():
        return "test" in cfg.purpose

    def checkASTExists():
        return "ast" in cfg.typesInput
    def checkASTSeqExists():
        return "astseq" in cfg.typesInput
    def checkCodeMetricsExists():
        return "codemetrics" in cfg.typesInput
    def checkCommitGraphExists():
        return "commitgraph" in cfg.typesInput
    def checkCommitSeqExists():
        return "commitseq" in cfg.typesInput
    def checkProcessMetricsExists():
        return "processmetrics" in cfg.typesInput