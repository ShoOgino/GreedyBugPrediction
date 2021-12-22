from src.config.config import config
from src.log.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.getPathFileLog())

import optuna
import numpy as np
import glob
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import json
import argparse
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
import pickle
from sklearn import svm
from sklearn.metrics import roc_auc_score

class ModelerRF():
    def __init__(self):
        self.trials4HyperParameterSearch = config.trials4HyperParameterSearch
        self.period4HyperParameterSearch = config.period4HyperParameterSearch

    def searchHyperParameter(self, datasets_Train_Valid):
        def objectiveFunction(trial):
            hp = {
                'max_depth': trial.suggest_int('max_depth', 2,  256),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2,  256),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 256),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 256),
                'random_state':42
            }
            scoreAverage=0
            for index4CrossValidation in range(len(datasets_Train_Valid)):
                dataset4Train = datasets_Train_Valid[index4CrossValidation]["train"]
                dataset4Valid = datasets_Train_Valid[index4CrossValidation]["valid"]
                xTrain = [sample["x"]["codemetrics"]+sample["x"]["processmetrics"] for sample in dataset4Train]
                yTrain = [sample["y"] for sample in dataset4Train]
                xValid = [sample["x"]["codemetrics"]+sample["x"]["processmetrics"] for sample in dataset4Valid]
                yValid = [sample["y"] for sample in dataset4Valid]
                model = RandomForestRegressor(n_estimators=10000, **hp)
                model.fit(xTrain, yTrain)
                score = mean_squared_error(yValid, model.predict(xValid))
                scoreAverage += score
            scoreAverage = scoreAverage / config.splitSize4CrossValidation
            return scoreAverage
        logger.info("hyperparameter search started")
        config.pathDatabaseOptuna = config.pathDatabaseOptuna or config.pathDirOutput + "/optuna.db"
        study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDatabaseOptuna, load_if_exists=True)
        study.optimize(objectiveFunction, timeout=config.period4HyperParameterSearch)

    def loadHyperparameter(self):
        if(config.hyperparameter):
            return config.hyperparameter
        elif(config.pathDatabaseOptuna):
            # optunaデータベースから、最適なハイパーパラメータを読み出す
            study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDirOutput + "/optuna.db", load_if_exists=True)
            return dict(study.best_params.items()^study.best_trial.user_attrs.items())
        else:
            logger.error("Hyperparameter can't be loaded. config.hyperparameter or config.pathDatabaseOptuna are not defined")
            raise Exception()

    def buildModel(self, datasets_Train_Test):
        logger.info("build Model")

        hp = self.loadHyperparameter()

        model = RandomForestRegressor(n_estimators=10000, **hp)
        dataset4Train = datasets_Train_Test["train"]
        xTrain = [sample["x"]["codemetrics"]+sample["x"]["processmetrics"] for sample in dataset4Train]
        yTrain = [sample["y"] for sample in dataset4Train]
        model.fit(xTrain,yTrain)

        # save parameter that seems to be the best
        config.pathParameters = os.path.join(config.pathDirOutput, "parameters")
        with open(config.pathParameters, mode='wb') as file:
            pickle.dump(model, file)

    def testModel(self, dataset4Test):
        logger.info("test model")

        IDRecord = [sample["id"] for sample in dataset4Test]
        xTest    = [sample["x"]["codemetrics"] + sample["x"]["processmetrics"] for sample in dataset4Test]
        yTest    = [sample["y"] for sample in dataset4Test]

        with open(config.pathParameters, mode='rb') as file:
            model = pickle.load(file)
        yPredicted = model.predict(xTest).flatten()

        # output prediction result
        IDRecord = [sample["id"] for sample in dataset4Test]
        resultTest = np.stack((IDRecord, yTest, yPredicted), axis=1)
        with open(config.pathDirOutput+"/prediction.csv", 'w', newline="") as file:
            csv.writer(file).writerows(resultTest)

        # output recall, precision, f-measure, AUC
        yPredicted = np.round(yPredicted, 0)
        report = classification_report(yTest, yPredicted, output_dict=True)
        report["AUC"] = roc_auc_score(yTest, yPredicted)
        with open(config.pathDirOutput+"/report.json", 'w') as file:
            json.dump(report, file, indent=4)

        # output confusion matrics
        cm = confusion_matrix(yTest, yPredicted)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.savefig(config.pathDirOutput+"/ConfusionMatrix.png")