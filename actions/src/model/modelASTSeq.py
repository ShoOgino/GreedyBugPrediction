from unicodedata import bidirectional
from numpy.core.fromnumeric import size
from numpy.distutils.lib2def import output_def
from torch.nn.modules import dropout
from torch.nn.modules.rnn import LSTM
from torch.utils import data
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
from collections import OrderedDict
from torch.utils.data import DataLoader, dataset
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

class ModelASTSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.trials4HyperParameterSearch = 100
        self.isCrossValidation = True
        self.device = "cuda:0"
        torch.backends.cudnn.enabled = False

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.dataset[0] = torch.tensor(dataset[0]).float()
            print(self.dataset[0].shape)
            self.dataset[1] = torch.tensor(dataset[1]).float()
            print(self.dataset[1].shape)
        def getNumFeatures(self):
            return len(self.dataset[1][0][0])
        def __len__(self):
            return len(self.dataset[0])
        def __getitem__(self, index):
            return self.dataset[1][index], self.dataset[0][index]

    def getModel(self, hp, numFeatures):
        numLayers = hp["numLayers"]
        sizeOutput = 1
        numParameters = hp["numParameters"]
        rateDropout = hp["rateDropout"]
        self.LSTM = nn.LSTM(
                input_size = numFeatures,
                hidden_size = numParameters,
                num_layers = numLayers,
                batch_first = True,
                dropout = rateDropout,
                bidirectional = True
            )
        self.linearOutput = nn.Linear(numParameters*2*numLayers, sizeOutput)
        self.activationOutput = nn.Sigmoid()
        def forward(x):
            _, (parametersHiddenBiLSTM, _) = self.LSTM(x)
            parametersHiddenBiLSTM = torch.cat(torch.split(parametersHiddenBiLSTM, 1), dim=2)
            y = self.linearOutput(parametersHiddenBiLSTM)
            y = self.activationOutput(y)
            return y
        self.forward = forward
        model = self.to(self.device)
        summary(
            model,
            input_size=(1, 1024, numFeatures),
            col_names=["output_size", "num_params"]
        )
        return model

    def getOptimizer(self, hp, model):
        nameOptimizer = hp["optimizer"]
        if nameOptimizer == 'adam':
            lrAdam = hp["lrAdam"]
            beta_1Adam = hp["beta1Adam"]
            beta_2Adam = hp["beta2Adam"]
            epsilonAdam = hp["epsilonAdam"]
            optimizer = torch.optim.Adam(model.parameters(), lr=lrAdam, betas=(beta_1Adam,beta_2Adam), eps=epsilonAdam)
        return optimizer

    def train_(self, dataLoader, model, lossFunction, optimizer, numEpochs):
        lossesTrain = []
        lossesValid = []
        accsTrain = []
        accsValid = []
        lossValidBest = 10000
        epochBestValid = 0
        for epoch in range(numEpochs):
            for phase in ["train","valid"]:
                if phase=="train":
                    model.train()
                elif phase=="valid":
                    model.eval()
                loss_sum=0
                corrects=0
                total=0
                with tqdm(total=len(dataLoader[phase]),unit="batch") as pbar:
                    pbar.set_description(f"Epoch[{epoch}/{numEpochs}]({phase})")
                    for xs, ys in dataLoader[phase]:
                        xs, ys = xs.to(self.device), ys.to(self.device)
                        ysPredicted=model(xs)
                        ysPredicted = ysPredicted.squeeze()
                        ys = ys.squeeze()
                        loss=lossFunction(ysPredicted, ys)

                        if phase=="train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        ysPredicted =  torch.round(ysPredicted)
                        corrects+=int((ysPredicted==ys).sum())
                        total+=xs.size(0)
                        accuracy = corrects/total
                        #loss関数で通してでてきたlossはCrossEntropyLossのreduction="mean"なので平均
                        #batch sizeをかけることで、batch全体での合計を今までのloss_sumに足し合わせる
                        loss_sum += float(loss) * xs.size(0)
                        running_loss = loss_sum/total
                        pbar.set_postfix({"loss":running_loss,"accuracy":accuracy })
                        pbar.update(1)
                if(phase == "train"):
                    lossesTrain.append(loss_sum/total)
                    accsTrain.append(corrects/total)
                if(phase == "valid"):
                    lossesValid.append(loss_sum/total)
                    accsValid.append(corrects/total)
                    if(loss_sum < lossValidBest):
                        print("update!")
                        lossValidBest = loss_sum
                        epochBestValid = epoch
            if(100<epoch-epochBestValid):
                break
        return epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid

    def setTrials4HyperParameterSearch(self, trials4HyperParameterSearch):
        self.trials4HyperParameterSearch = trials4HyperParameterSearch

    def setPeriod4HyperParameterSearch(self, period4HyperPrameterSearch):
        self.period4HyperParameterSearch = period4HyperPrameterSearch

    def setIsCrossValidation(self, isCrossValidation):
        self.isCrossValidation = isCrossValidation

    def plotLearningCurve(self, lossesTrain, lossesValid, accTrain, accValid, numberTrial):
        epochs = range(len(lossesTrain))

        fig = plt.figure()
        plt.ylim(0, 2)
        plt.plot(epochs, lossesTrain, linestyle="-", color='b', label = 'lossTrain')
        plt.plot(epochs, accTrain, linestyle="-", color='r', label = 'accTrain')
        plt.plot(epochs, lossesValid, linestyle=":", color='b' , label= 'lossVal')
        plt.plot(epochs, accValid, linestyle=":", color='r' , label= 'accVal')
        plt.title(str(numberTrial))
        plt.legend()

        pathLogGraph = os.path.join(Result4BugPrediction.getPathResult(), str(numberTrial) + '.png')
        fig.savefig(pathLogGraph)
        plt.clf()
        plt.close()

    def searchHyperParameter(self, arrayOfD4TAndD4V):
        def objectiveFunction(trial):
            hp = {}
            hp["numLayers"]=trial.suggest_int('numlayers', 1, 1)
            hp["numParameters"] = int(trial.suggest_int('numParameters', 16, 512))
            hp["rateDropout"] = trial.suggest_uniform('rateDropout', 0.0, 0.5)
            hp["optimizer"] = trial.suggest_categorical('optimizer', ['adam'])
            hp["lrAdam"] = trial.suggest_loguniform('lrAdam', 1e-5, 1e-2)
            hp["beta1Adam"] = trial.suggest_uniform('beta1Adam', 0.9, 1)
            hp["beta2Adam"] = trial.suggest_uniform('beta2Adam', 0.999, 1)
            hp["epsilonAdam"] = trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-5)
            hp["sizeBatch"] = trial.suggest_int("sizeBatch", 16, 128)
            numEpochs = 10000
            scoreAverage=0
            for index4CrossValidation in range(len(arrayOfD4TAndD4V)):
                # prepare dataset
                dataset4Train = self.Dataset([list(i) for i in zip(*arrayOfD4TAndD4V[index4CrossValidation]["training"])][1:])
                dataset4Test = self.Dataset([list(i) for i in zip(*arrayOfD4TAndD4V[index4CrossValidation]["validation"])][1:])
                dataloader={
                    "train": DataLoader(dataset4Train, batch_size = hp["sizeBatch"], pin_memory=True),
                    "valid": DataLoader(dataset4Test, batch_size = hp["sizeBatch"], pin_memory=True)
                }

                # prepare model architecture
                model = self.getModel(hp, dataset4Train.getNumFeatures())

                # prepare loss function
                lossFunction = nn.BCELoss()

                # prepare  optimizer
                optimizer = self.getOptimizer(hp, model)

                # train!
                epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid = self.train_(dataloader, model, lossFunction, optimizer, numEpochs)
                self.plotLearningCurve(lossesTrain, lossesValid, accsTrain, accsValid, trial.number)
                trial.set_user_attr("numEpochs", epochBestValid)
                # 1エポックだけ偶然高い精度が出たような場合を弾くために、前後のepochで平均を取る。
                lossValMin = min(lossesValid)
                indexValMin = lossesValid.index(lossValMin)
                indexLast = len(lossesValid)-1
                index5Forward = indexValMin+5 if indexValMin+5 < indexLast else indexLast
                score=0
                for i in range(6):
                    score += lossesValid[index5Forward-i]
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

    def searchParameter(self, dataset4SearchParameter):
        with open(Result4BugPrediction.getPathHyperParameter(), mode='r') as file:
            hp = json.load(file)

        # prepare dataset
        dataset4Train = self.Dataset([list(i) for i in zip(*dataset4SearchParameter["train"])][1:])
        dataset4Test = self.Dataset([list(i) for i in zip(*dataset4SearchParameter["valid"])][1:])
        dataloader={
            "train": DataLoader(dataset4Train, batch_size = hp["sizeBatch"], pin_memory=True),
            "valid": DataLoader(dataset4Test, batch_size = hp["sizeBatch"], pin_memory=True)
        }

        # prepare model architecture
        model = self.getModel(hp, dataset4Train.getNumFeatures())

        # prepare loss function
        lossFunction = nn.BCELoss()

        # prepare  optimizer
        optimizer = self.getOptimizer(hp, model)

        # prepare model parameters
        _, lossesTrain, lossesValid, accsTrain, accsValid = self.train_(dataloader, model, lossFunction, optimizer, hp["numEpochs"])
        self.plotLearningCurve(lossesTrain, lossesValid, accsTrain, accsValid, 10000)

        pathParameter = os.path.join(Result4BugPrediction.getPathResult(), 'parameter')
        torch.save(model.state_dict(), pathParameter)
        return pathParameter

    def test(self, dataset4Test):
        with open(Result4BugPrediction.getPathHyperParameter(), mode='r') as file:
            hp = json.load(file)

        IDRecord = [list(i) for i in zip(*dataset4Test)][0]
        # prepare dataset
        dataset4Test = self.Dataset([list(i) for i in zip(*dataset4Test)][1:])
        dataloader={
            "valid": DataLoader(dataset4Test, batch_size = hp["sizeBatch"], pin_memory=True)
        }

        # prepare model architecture
        model = self.getModel(hp, dataset4Test.getNumFeatures())

        # prepare model parameters
        paramaters = torch.load(Result4BugPrediction.getPathParameter())
        model.load_state_dict(paramaters)
        model = model.eval()

        # predict ys
        yPredicted = []
        yTest = []
        for xs, ys in dataloader["valid"]:
            xs = xs.to(self.device)
            with torch.no_grad():
                output = model(xs)
                yPredicted += [l for l in output.to("cpu").squeeze().tolist()]
                yTest += [l for l in ys.to("cpu").squeeze().tolist()]

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