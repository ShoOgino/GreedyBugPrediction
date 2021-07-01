from src.dataset.datasets import Datasets
from src.model.model import Model
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

        self.dataset = Datasets()
        # "ast", "astseq", "codemetrics", "commitgraph", "commitseq", "processmetrics"
        self.dataset.setInputData(["ast", "astseq", "codemetrics", "commitgraph", "commitseq", "processmetrics"])
        self.dataset.setPathsSample(
            [
                r"C:\Users\login\data\workspace\MLTool\datasets\egit\output\R5_r_train"
            ],
            isForTest = False
        )
        self.dataset.setPathsSample(
            [
                r"C:\Users\login\data\workspace\MLTool\datasets\egit\output\R5_r_test"
            ], 
            isForTest = True
        )
        self.model = Model()
        self.model.setIsCrossValidation(False)
        self.model.setSplitSize4Validation(5)
        self.model.setPeriod4HyperParameterSearch(60*60*10)

        Result4BugPrediction.clear()
        Result4BugPrediction.setPathResult(os.path.dirname(os.path.dirname(__file__))+"/results/"+self.id)
        os.makedirs(Result4BugPrediction.pathResult, exist_ok=True)
        shutil.copy(__file__, Result4BugPrediction.pathResult)

experiment = Experiment()
maneger = Maneger()
maneger.run(experiment)