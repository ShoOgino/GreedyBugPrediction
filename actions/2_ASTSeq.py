from src.dataset.datasetASTSeq import DatasetASTSeq
from src.model.modelASTSeq import ModelASTSeq
from src.result.result4BugPrediction import Result4BugPrediction
from src.manager import Maneger

import os
import glob
import shutil
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing

class Experiment():
    def __init__(self):
        self.id = "ASTSeq"

        self.purpose = ["searchHyperParameter", "searchParameter", "test"]

        self.dataset = DatasetASTSeq()
        self.dataset.setPathSamples(
            [
                r"C:\Users\login\data\workspace\GBP\datasets\egit\modules_"
            ]
        )
        self.dataset.setIsCrossValidation(False)
        self.dataset.setSplitSize4Validation(5)

        self.model = ModelASTSeq()
        self.model.setTrials4HyperParameterSearch(1)

        Result4BugPrediction.clear()
        Result4BugPrediction.setPathResult(os.path.dirname(os.path.dirname(__file__))+"/results/"+self.id)
        os.makedirs(Result4BugPrediction.pathResult, exist_ok=True)
        shutil.copy(__file__, Result4BugPrediction.pathResult)

experiment = Experiment()
maneger = Maneger()
maneger.run(experiment)