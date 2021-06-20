from src.result.result4BugPrediction import Result4BugPrediction
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import optuna
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pickle
import tensorflow
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, concatenate, Activation, Flatten, Reshape, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping

class DNN4BugPrediction(nn.Module):
    def __init__(self):
        self.trials4HyperParameterSearch = 100
        self.isCrossValidation = True
        self.setProcessor()

    def setTrials4HyperParameterSearch(self, trials4HyperParameterSearch):
        self.trials4HyperParameterSearch = trials4HyperParameterSearch

    def setIsCrossValidation(self, isCrossValidation):
        self.isCrossValidation = isCrossValidation

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

        pathLogGraph = os.path.join(Result4BugPrediction.getPathResult(), str(numberTrial) + '.png')
        fig.savefig(pathLogGraph)
        plt.clf()
        plt.close()


    def searchHyperParameter(self, arrayOfD4TAndD4V):
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
                    optimizer = optimizers.Adam(learning_rate=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
                return optimizer
            scoreAverage=0
            for i in range(len(arrayOfD4TAndD4V)):
                xTrain = np.array([list(i) for i in zip(*arrayOfD4TAndD4V[i]["training"])][2])
                yTrain = np.array([list(i) for i in zip(*arrayOfD4TAndD4V[i]["training"])][1])
                xValid = np.array([list(i) for i in zip(*arrayOfD4TAndD4V[i]["validation"])][2])
                yValid = np.array([list(i) for i in zip(*arrayOfD4TAndD4V[i]["validation"])][1])
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
            scoreAverage = scoreAverage / len(arrayOfD4TAndD4V)
            #全体のログをloggerで出力
            with open(Result4BugPrediction.getPathLogSearchHyperParameter(), mode='a') as f:
                f.write(str(score)+","+str(trial.datetime_start)+","+str(trial.params)+'\n')
            return scoreAverage
        study = optuna.create_study()
        study.optimize(objectiveFunction, n_trials=self.trials4HyperParameterSearch)
        #save the hyperparameter that seems to be the best.
        with open(Result4BugPrediction.getPathHyperParameter(), mode='a') as file:
            json.dump(dict(study.best_params.items()^study.best_trial.user_attrs.items()), file, indent=4)
        return Result4BugPrediction.getPathHyperParameter()

    def searchParameter(self, dataset4Training):
        xTrain = np.array([list(i) for i in zip(*dataset4Training)][2])
        yTrain = np.array([list(i) for i in zip(*dataset4Training)][1])
        with open(Result4BugPrediction.getPathHyperParameter(), mode='r') as file:
            hp = json.load(file)
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
            opt = keras.optimizers.SGD(learning_rate=lrSgd, momentum=momentumSgd)
        elif hp["optimizer"] == "adagrad":
            opt = keras.optimizers.Adagrad()
        elif hp["optimizer"] == "adadelta":
            opt = keras.optimizers.Adadelta()
        elif hp["optimizer"] == 'adam':
            lrAdam = hp["lrAdam"]
            beta_1Adam = hp["beta1Adam"]
            beta_2Adam = hp["beta2Adam"]
            epsilonAdam = hp["epsilonAdam"]
            opt = optimizers.Adam(learning_rate=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
        elif hp["optimizer"] == "adamax":
            lrAdamax = hp["lrAdamax"]
            beta_1Adamax = hp["beta1Adamax"]
            beta_2Adamax = hp["beta2Adamax"]
            epsilonAdamax = hp["epsilonAdamax"]
            opt = keras.optimizers.Adamax(learning_rate=lrAdamax, beta_1=beta_1Adamax, beta_2=beta_2Adamax, epsilon=epsilonAdamax)
        elif hp["optimizer"] == "nadam":
            lrNadam = hp["lrNadam"]
            beta_1Nadam = hp["beta1Nadam"]
            beta_2Nadam = hp["beta2Nadam"]
            epsilonNadam = hp["epsilonNadam"]
            opt = optimizers.Nadam(learning_rate=lrNadam, beta_1=beta_1Nadam, beta_2=beta_2Nadam, epsilon=epsilonNadam)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        xTrain = np.array([list(i) for i in zip(*dataset4Training)][2])
        yTrain = np.array([list(i) for i in zip(*dataset4Training)][1])
        model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose)

        # save parameter that seems to be the best
        pathParameter = os.path.join(Result4BugPrediction.getPathResult(), 'parameter')
        model.save(pathParameter)
        return pathParameter

    def test(self, dataset4Test):
        IDRecord = [list(i) for i in zip(*dataset4Test)][0]
        xTest = np.array([list(i) for i in zip(*dataset4Test)][2])
        yTest = np.array([list(i) for i in zip(*dataset4Test)][1])

        model = load_model(Result4BugPrediction.getPathParameter())
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

    def setProcessor(self):
        physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tensorflow.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tensorflow.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
