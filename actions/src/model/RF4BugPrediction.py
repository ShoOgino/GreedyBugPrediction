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
from src.result.result4BugPrediction import Result4BugPrediction

class RF4BugPrediction():
    def __init__(self):
        self.period4HyperParameterSearch = 60*1
        self.trials4HyperParameterSearch = 100
        self.isCrossValidation = True

    def setTrials4HyperParameterSearch(self, trials4HyperParameterSearch):
        self.trials4HyperParameterSearch = trials4HyperParameterSearch

    def setPeriod4HyperParameterSearch(self, period4HyperParameterSearch):
        self.period4HyperParameterSearch = period4HyperParameterSearch

    def setIsCrossValidation(self, isCrossValidation):
        self.isCrossValidation = isCrossValidation

    def searchHyperParameter(self, arrayOfD4TAndD4V):
        def objectiveFunction(trial):
            hp = {
                'n_estimators': trial.suggest_int('n_estimators', 2, 256),
                'max_depth': trial.suggest_int('max_depth', 2,  256),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2,  256),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 256),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 256),
                'random_state':42
            }
            scoreAverage=0
            for i in range(len(arrayOfD4TAndD4V)):
                xTrain = [list(i) for i in zip(*arrayOfD4TAndD4V[i]["training"])][2]
                yTrain = [list(i) for i in zip(*arrayOfD4TAndD4V[i]["training"])][1]
                xValid = [list(i) for i in zip(*arrayOfD4TAndD4V[i]["validation"])][2]
                yValid = [list(i) for i in zip(*arrayOfD4TAndD4V[i]["validation"])][1]
                model = RandomForestRegressor(**hp)
                model.fit(xTrain, yTrain)
                score = mean_squared_error(yValid, model.predict(xValid))
                scoreAverage += score
            scoreAverage = scoreAverage / len(arrayOfD4TAndD4V)
            #全体のログをloggerで出力
            with open(Result4BugPrediction.getPathLogSearchHyperParameter(), mode='a') as f:
                f.write(str(scoreAverage)+","+str(trial.datetime_start)+","+str(trial.params)+'\n')
            return scoreAverage
        optuna.logging.disable_default_handler()
        study = optuna.create_study()
        study.optimize(objectiveFunction, timeout=self.period4HyperParameterSearch)#n_trials=self.trials4HyperParameterSearch)

        #save the hyperparameter that seems to be the best.
        with open(Result4BugPrediction.getPathHyperParameter(), mode='w') as file:
            json.dump(dict(study.best_params.items()^study.best_trial.user_attrs.items()), file, indent=4)
        return Result4BugPrediction.getPathHyperParameter()

    def searchParameter(self, dataset4Training):
        with open(Result4BugPrediction.getPathHyperParameter(), mode='r') as file:
            print(Result4BugPrediction.getPathHyperParameter())
            hp = json.load(file)
        model=RandomForestRegressor(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            max_leaf_nodes=hp["max_leaf_nodes"],
            min_samples_leaf=hp["min_samples_leaf"],
            min_samples_split=hp["min_samples_split"],
            random_state=42)
        xTrain = [list(i) for i in zip(*dataset4Training)][2]
        yTrain = [list(i) for i in zip(*dataset4Training)][1]
        model.fit(xTrain,yTrain)

        # save parameter that seems to be the best
        with open(Result4BugPrediction.getPathParameter(), mode='wb') as file:
            pickle.dump(model, file)
        return Result4BugPrediction.getPathParameter()

    def test(self, dataset4Test):
        IDRecord = [list(i) for i in zip(*dataset4Test)][0]
        xTest = [list(i) for i in zip(*dataset4Test)][2]
        yTest = [list(i) for i in zip(*dataset4Test)][1]

        with open(Result4BugPrediction.getPathParameter(), mode='rb') as file:
            model = pickle.load(file)
        yPredicted = model.predict(xTest).flatten()

        # output prediction result
        resultTest = np.stack((IDRecord, yTest, yPredicted), axis=1)
        with open(Result4BugPrediction.pathResult+"/prediction.csv", 'w', newline="") as file:
            csv.writer(file).writerows(resultTest)

        # output recall, precision, f-measure, AUC
        yPredicted = np.round(yPredicted, 0)
        report = classification_report(yTest, yPredicted, output_dict=True)
        report["AUC"] = roc_auc_score(yTest, yPredicted)
        with open(Result4BugPrediction.pathResult+"/report.json", 'w') as file:
            json.dump(report, file, indent=4)

        # output confusion matrics
        cm = confusion_matrix(yTest, yPredicted)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.savefig(Result4BugPrediction.pathResult+"/ConfusionMatrix.png")
