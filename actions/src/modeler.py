import optuna
from keras.layers import Input, concatenate
from keras.layers.core import Activation, Flatten, Reshape, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import Model
import pandas as pd
import numpy as np
import glob
import os
import csv
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.keras import TqdmCallback
import sys
import tensorflow as tf
import json
import argparse
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
import pickle
from sklearn import svm
from sklearn.metrics import roc_auc_score

class Modeler:
    def __init__(self, dirResults):
        self.dirResults = dirResults

    def plotLearningCurve(self, history, numberTrial):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        fig = plt.figure()
        plt.ylim(0, 2)
        plt.plot(epochs, loss, linestyle="-", color='r', label = 'lossTrain')
        plt.plot(epochs, acc, linestyle="-", color='b', label = 'accTrain')
        plt.plot(epochs, val_loss, linestyle=":", color='r' , label= 'lossVal')
        plt.plot(epochs, val_acc, linestyle=":", color='b' , label= 'accVal')
        plt.title(str(numberTrial))
        plt.legend()

        pathLogGraph = os.path.join(self.dirResults, str(numberTrial) + '.png')
        fig.savefig(pathLogGraph)
        plt.clf()
        plt.close()

    def searchHyperParameter(self, dataset4tuningHP, modelAlgorithm, NOfTrials):
        if modelAlgorithm=="SVM":
            def objectiveFunction(trial):
                hp={
                    "kernel" : trial.suggest_categorical('kernel', ["linear"])
                }
                if hp["kernel"] == 'linear':
                    hp["C"] = trial.suggest_int("C", 1, 1000)
                if hp["kernel"] == 'rbf':
                    hp["C"] = trial.suggest_int("C", 1, 1000)
                    hp["gamma"] = trial.suggest_categorical("gammma", ["scale", "auto"])
                if hp["kernel"] == 'poly':
                    hp["C"] = trial.suggest_int("C", 1, 1000)
                    hp["gamma"] = trial.suggest_categorical("gammma", ["scale", "auto"])
                    hp["degree"] = trial.suggest_int("degree", 2, 4)
                if hp["kernel"] == 'sigmoid':
                    hp["C"] = trial.suggest_int("C", 1, 1000)
                    hp["gamma"] = trial.suggest_categorical("gammma", ["scale", "auto"])
                print(hp)
                scoreAverage=0
                for i in range(len(dataset4tuningHP)):
                    xTrain = dataset4tuningHP[i]["xTrain"]
                    yTrain = dataset4tuningHP[i]["yTrain"]
                    xValid = dataset4tuningHP[i]["xValid"]
                    yValid = dataset4tuningHP[i]["yValid"]
                    model = svm.SVC(**hp)
                    print(len(xTrain))
                    print(len(yTrain))
                    model.fit(xTrain, yTrain)
                    score = mean_squared_error(yValid, model.predict(xValid))
                    scoreAverage += score
                scoreAverage = scoreAverage / len(dataset4tuningHP)
                with open(os.path.join(self.dirResults, "results.txt"), mode='a') as f:
                    f.write(str(score)+","+str(trial.datetime_start)+","+str(trial.params)+'\n')
                return scoreAverage
        elif modelAlgorithm=="RF":
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
                for i in range(len(dataset4tuningHP)):
                    xTrain = dataset4tuningHP[i]["xTrain"]
                    yTrain = dataset4tuningHP[i]["yTrain"]
                    xValid = dataset4tuningHP[i]["xValid"]
                    yValid = dataset4tuningHP[i]["yValid"]
                    model = RandomForestRegressor(**hp)
                    model.fit(xTrain, yTrain)
                    score = mean_squared_error(yValid, model.predict(xValid))
                    scoreAverage += score
                    print(score)
                scoreAverage = scoreAverage / len(dataset4tuningHP)
                #全体のログをtxt形式で出力
                with open(os.path.join(self.dirResults, "results.txt"), mode='a') as f:
                    f.write(str(score)+","+str(trial.datetime_start)+","+str(trial.params)+'\n')
                return scoreAverage
        elif modelAlgorithm=="DNN":
            def objectiveFunction(trial):
                def chooseModel(trial):
                    n_outputs = 1
                    NOLayers = trial.suggest_int('NOlayers', 1, 3)
                    model = Sequential()
                    for i in range(NOLayers):
                        NOUnits=int(trial.suggest_int('NOUnits{}'.format(i), 16, 1024))
                        rateDropout = trial.suggest_uniform('rateDropout{}'.format(i), 0.0, 0.5)
                        activation = trial.suggest_categorical('activation{}'.format(i), ['hard_sigmoid', 'linear', 'relu', 'sigmoid', 'softplus','softsign', 'tanh'])
                        model.add(Dense(NOUnits, activation=activation))
                        model.add(Dropout(rateDropout))
                    model.add(Dense(n_outputs,activation='sigmoid'))
                    return model
                def chooseOptimizer(trial):
                    nameOptimizer = trial.suggest_categorical('optimizer', ['adam'])
                    if nameOptimizer == 'adam':
                        lrAdam = trial.suggest_loguniform('lrAdam', 1e-5, 1e-2)
                        beta_1Adam = trial.suggest_uniform('beta1Adam', 0.9, 1)
                        beta_2Adam = trial.suggest_uniform('beta2Adam', 0.999, 1)
                        epsilonAdam = trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-5)
                        opt = keras.optimizers.Adam(lr=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
                    return opt
                scoreAverage=0
                for i in range(len(dataset4tuningHP)):
                    xTrain = dataset4tuningHP[i]["xTrain"]
                    yTrain = dataset4tuningHP[i]["yTrain"]
                    xValid = dataset4tuningHP[i]["xValid"]
                    yValid = dataset4tuningHP[i]["yValid"]
                    verbose, epochs, sizeBatch = 0,  10000, trial.suggest_int("sizeBatch", 64, 1024)
                    numFeatures = xTrain.shape[1]
                    model = chooseModel(trial)
                    opt = chooseOptimizer(trial)
                    model.build((None,numFeatures))
                    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
                    epochsEarlyStop=100
                    history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose, validation_data=(xValid, yValid), callbacks=[EarlyStopping(monitor='val_loss', patience=epochsEarlyStop, mode='auto')])
                    self.plotLearningCurve(history, trial.number)
                    trial.set_user_attr("epochs", len(history.history['val_loss'])-100)
                    # 1エポックだけ偶然高い精度が出たような場合を弾く。
                    lossesVal = history.history['val_loss']
                    lossValMin = min(lossesVal)
                    indexValMin = lossesVal.index(lossValMin)
                    indexLast = len(lossesVal)-1
                    index5Forward = indexValMin+5 if indexValMin+5 < indexLast else indexLast
                    score=0
                    for i in range(6):
                        score += lossesVal[index5Forward-i]
                    score = score / 6
                    scoreAverage += score
                scoreAverage = scoreAverage / len(dataset4tuningHP)
                #ログをtxt形式で出力
                with open(os.path.join(self.dirResults, "results.txt"), mode='a') as f:
                    f.write(str(score)+","+str(trial.datetime_start)+","+str(trial.params)+'\n')
                return scoreAverage
        study = optuna.create_study()
        study.optimize(objectiveFunction, n_trials=NOfTrials)

        # save the hyperparameter that seems to be the best.
        pathHyperParameter = os.path.join(self.dirResults, "hyperparameter.json")
        with open(pathHyperParameter, mode='a') as file:
            json.dump(dict(study.best_params.items()^study.best_trial.user_attrs.items()), file, indent=4)
        return pathHyperParameter

    def searchParameter(self, xTrain, yTrain, xTest, yTest, modelAlgorithm, pathHP):
        with open(pathHP, mode='r') as file:
            hp = json.load(file)
        if modelAlgorithm=="SVM":
            model = SVM(**hp)
            model.fit(xTrain, yTrain)
            # save parameter that seems to be the best
            pathParameter = os.path.join(self.dirResults, 'parameter')
            with open(pathParameter, mode='wb') as file:
                pickle.dump(model, file)
            return pathParameter
        elif modelAlgorithm=="RF":
            model=RandomForestRegressor(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                max_leaf_nodes=hp["max_leaf_nodes"],
                min_samples_leaf=hp["min_samples_leaf"],
                min_samples_split=hp["min_samples_split"],
                random_state=42)
            model.fit(xTrain,yTrain)

            # save parameter that seems to be the best
            pathParameter = os.path.join(self.dirResults, 'parameter')
            with open(pathParameter, mode='wb') as file:
                pickle.dump(model, file)
            return pathParameter
        elif modelAlgorithm=="DNN":
            n_outputs = 1
            model = Sequential()
            for i in range(int(hp["NOlayers"])):
                model.add(Dense(hp["NOUnits"+str(i)], activation=hp["activation"+str(i)]))
                model.add(Dropout(hp["rateDropout"+str(i)]))
                model.add(Dense(n_outputs,activation='sigmoid'))
            verbose, epochs, sizeBatch = 1, hp["epochs"], hp["sizeBatch"]
            n_features, n_outputs = xTrain.shape[1], 1
            model.build((None,n_features))
            if hp["optimizer"] == "sgd":
                lrSgd = hp["lrSgd"]
                momentumSgd = hp["momentumSgd"]
                opt = keras.optimizers.SGD(lr=lrSgd, momentum=momentumSgd)
            elif hp["optimizer"] == "adagrad":
                opt = keras.optimizers.Adagrad()
            elif hp["optimizer"] == "adadelta":
                opt = keras.optimizers.Adadelta()
            elif hp["optimizer"] == 'adam':
                lrAdam = hp["lrAdam"]
                beta_1Adam = hp["beta1Adam"]
                beta_2Adam = hp["beta2Adam"]
                epsilonAdam = hp["epsilonAdam"]
                opt = keras.optimizers.Adam(lr=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
            elif hp["optimizer"] == "adamax":
                lrAdamax = hp["lrAdamax"]
                beta_1Adamax = hp["beta1Adamax"]
                beta_2Adamax = hp["beta2Adamax"]
                epsilonAdamax = hp["epsilonAdamax"]
                opt = keras.optimizers.Adamax(lr=lrAdamax, beta_1=beta_1Adamax, beta_2=beta_2Adamax, epsilon=epsilonAdamax)
            elif hp["optimizer"] == "nadam":
                lrNadam = hp["lrNadam"]
                beta_1Nadam = hp["beta1Nadam"]
                beta_2Nadam = hp["beta2Nadam"]
                epsilonNadam = hp["epsilonNadam"]
                opt = keras.optimizers.Nadam(lr=lrNadam, beta_1=beta_1Nadam, beta_2=beta_2Nadam, epsilon=epsilonNadam)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
            model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose, validation_data=(xTest, yTest))

            # save parameter that seems to be the best
            pathParameter = os.path.join(self.dirResults, 'parameter')
            model.save(pathParameter)
            return pathParameter
        else:
            raise Exception("modelAlgorithm must be RF or DNN")

    def test(self, IDRecord, xTest, yTest, modelAlgorithm, pathParameter):
        if modelAlgorithm=="SVM":
            with open(pathParameter, mode='rb') as file:
                model = pickle.load(file)
            yPredicted = model.predict(xTest).flatten()
        elif modelAlgorithm=="RF":
            with open(pathParameter, mode='rb') as file:
                model = pickle.load(file)
            yPredicted=model.predict(xTest).flatten()
        elif modelAlgorithm=="DNN":
            model = keras.models.load_model(pathParameter)
            yPredicted = model.predict(xTest).flatten()
        else:
            raise Exception("modelAlgorithm must be RF or DNN")
        # output prediction result
        resultTest=np.stack((IDRecord, yTest, yPredicted), axis=1)
        pathResultTest=os.path.join(self.dirResults, "resultPrediction.csv")
        with open(pathResultTest, 'w', newline="") as file:
            csv.writer(file).writerows(resultTest)

        # output recall, precision, f-measure, AUC
        yPredicted = np.round(yPredicted, 0)
        report = classification_report(yTest, yPredicted, output_dict=True)
        report["AUC"] = roc_auc_score(yTest, yPredicted)
        pathResultReport = os.path.join(self.dirResults, "report.json")
        with open(pathResultReport, 'w') as file:
            json.dump(report, file, indent=4)

        # output confusion matrics
        cm = confusion_matrix(yTest, yPredicted)
        sns.heatmap(cm, annot=True, cmap='Blues')
        pathConfusionMatrix=os.path.join(self.dirResults, "confusionMatrix.png")
        plt.savefig(pathConfusionMatrix)