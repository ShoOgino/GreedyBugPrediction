from enum import Enum
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing

class config:
    # general
    id = None
    purpose = None
    pathDirOutput = None
    # load dataset
    typesInput = None
    pathsDirSampleTrain = None
    pathsDirSampleTest = None
    # build model 
    device = None
    epochs4EarlyStopping = None
    # tune hyperparameter
    isCrossValidation = None
    splitSize4CrossValidation = None
    trials4HyperParameterSearch = None
    period4HyperParameterSearch = None
    # load hyperparameter
    pathDatabaseOptuna = None
    hyperparameter = None
    # use model
    pathModel = None

    def getDirOutput():
        return config.pathDirOutput
    def getPathFileLog():
        return config.pathDirOutput+"/log.txt"

    class Purpose(Enum):
        searchHyperParameter = 1
        buildModel = 2
        testModel = 3
    class TypeInput(Enum):
        mnist4test = 0
        ast = 1
        astseq = 2
        codemetrics = 3
        commitgraph = 4
        commitseq = 5
        processmetrics = 6

    def checkPurposeContainsSearchHyperParameter():
        return config.Purpose.searchHyperParameter in config.purpose
    def checkPurposeContainsBuildModel():
        return config.Purpose.buildModel in config.purpose
    def checkPurposeContainsTestModel():
        return config.Purpose.testModel in config.purpose

    def checkMnist4TestExists():
        return config.TypeInput.mnist4test in config.typesInput
    def checkASTExists():
        return config.TypeInput.ast in config.typesInput
    def checkASTSeqExists():
        return config.TypeInput.astseq in config.typesInput
    def checkCodeMetricsExists():
        return config.TypeInput.codemetrics in config.typesInput
    def checkCommitGraphExists():
        return config.TypeInput.commitgraph in config.typesInput
    def checkCommitSeqExists():
        return config.TypeInput.commitseq in config.typesInput
    def checkProcessMetricsExists():
        return config.TypeInput.processmetrics in config.typesInput