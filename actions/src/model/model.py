from unicodedata import bidirectional
from numpy.core.fromnumeric import size
from numpy.distutils.lib2def import output_def
from torch.nn.modules import dropout
from torch.nn.modules.rnn import LSTM
from torch.utils import data
from src.result.result4BugPrediction import Result4BugPrediction
import torch
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
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm
from src.dataset.dataset import Dataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.trials4HyperParameterSearch = 100
        self.epochsEarlyStopping = 100
        self.isCrossValidation = True
        self.splitsize = 5
        self.device = "cuda:0"
        torch.backends.cudnn.enabled = True

    def setTrials4HyperParameterSearch(self, trials4HyperParameterSearch):
        self.trials4HyperParameterSearch = trials4HyperParameterSearch

    def setPeriod4HyperParameterSearch(self, period4HyperPrameterSearch):
        self.period4HyperParameterSearch = period4HyperPrameterSearch

    def setIsCrossValidation(self, isCrossValidation):
        self.isCrossValidation = isCrossValidation

    def setSplitSize4Validation(self, splitsize):
        self.splitsize = splitsize

    def defineNetwork(self, hp, dataset):
        # todo ここで、datasetの説明変数にどんな構造のデータが何種類あるのかを把握。それ次第でモデルの形状を変更する。
        self.componentsNetwork = {}
        if(0 < len(dataset.samples[0]["x"]["ast"]["nodes"])):
            pass
        if(0 < len(dataset.samples[0]["x"]["astseq"])):
            self.componentsNetwork["astseq"]["lstm"] = nn.LSTM(
                    input_size = 99,
                    hidden_size = hp["numParameters"],
                    num_layers = hp["numLayers"],
                    batch_first = True,
                    dropout = hp["rateDropout"],
                    bidirectional = True
                )
            self.componentsNetwork["astseq"]["linear"] = nn.Linear(hp["numParameters"]*2*hp["numLayers"], len(dataset.samples[0]["y"]))
        if(0 < len(dataset.samples[0]["x"]["codemetrics"])):
            pass
        if(0 < len(dataset.samples[0]["x"]["commitgraph"]["nodes"])):
            pass
        if(0 < len(dataset.samples[0]["x"]["commitseq"])):
            pass
        if(0 < len(dataset.samples[0]["x"]["processmetrics"])):
            pass
        self.componentsNetwork["activation"] = nn.Sigmoid()
        def forward(x):
            if("ast" in x):
                pass
            if("astseq" in x):
                _, (parametersHiddenBiLSTM, _) = self.componentsNetwork["astseq"]["LSTM"](x)
                parametersHiddenBiLSTM = torch.cat(torch.split(parametersHiddenBiLSTM, 1), dim=2)
                astseq = self.componentsNetwork["astseq"]["linear"](parametersHiddenBiLSTM)
            if("codemetrics" in x):
                pass
            if("commitgraph" in x):
                pass
            if("commitseq" in x):
                pass
            if("processmetrics" in x):
                pass
            y = self.componentsNetwork["activation"](astseq)
            return y
        self.forward = forward
        model = self.to(self.device)
        summary(
            model,
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
                        total+=ys.size(0)
                        accuracy = corrects/total
                        #loss関数で通してでてきたlossはCrossEntropyLossのreduction="mean"なので平均
                        #batch sizeをかけることで、batch全体での合計を今までのloss_sumに足し合わせる
                        loss_sum += float(loss) * ys.size(0)
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
            if(self.epochsEarlyStopping<epoch-epochBestValid):
                break
        return epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid


    def plotGraphTraining(self, lossesTrain, lossesValid, accTrain, accValid, numberTrial):
        epochs = range(len(lossesTrain))

        fig = plt.figure()
        plt.ylim(0, 2)
        plt.plot(epochs, lossesTrain, linestyle="-", color='b', label = 'lossTrain')
        plt.plot(epochs, accTrain, linestyle="-", color='r', label = 'accTrain')
        plt.plot(epochs, lossesValid, linestyle=":", color='b' , label= 'lossVal')
        plt.plot(epochs, accValid, linestyle=":", color='r' , label= 'accVal')
        plt.title(str(numberTrial))
        plt.legend()

        pathGraph = os.path.join(Result4BugPrediction.getPathResult(), str(numberTrial) + '.png')
        fig.savefig(pathGraph)
        plt.clf()
        plt.close()

    def plotGraphHyperParameterSearch(self, trials):
        numOfTrials = range(len(trials))

        fig = plt.figure()
        plt.title("HyperParameterSearch")
        plt.ylim(0, 1)
        plt.plot(numOfTrials, trials, linestyle="-", color='b', label = 'lossTrain')
        plt.legend()

        pathGraph = os.path.join(Result4BugPrediction.getPathResult(), str("hyperParameterSearch") + '.png')
        fig.savefig(pathGraph)
        plt.clf()
        plt.close()

    def searchHyperParameter(self, datasets_Train_Valid):
        def objectiveFunction(trial):
            numEpochs = 10000
            scoreAverage=0
            for index4CrossValidation in range(len(datasets_Train_Valid)):
                # prepare dataset
                dataset4Train = Dataset(datasets_Train_Valid[index4CrossValidation]["train"])
                dataset4Test = Dataset(datasets_Train_Valid[index4CrossValidation]["valid"])
                hp = {}
                if(0 < len(dataset4Train.samples[0]["x"]["ast"]["nodes"])):
                    hp["ast"] = {}
                if(0 < len(dataset4Train.samples[0]["x"]["astseq"])):
                    hp["astseq"] = {}
                    hp["astseq"]["numLayers"] = trial.suggest_int('numLayers', 1, 3)
                    hp["astseq"]["numParameters"] = int(trial.suggest_int('numParameters', 16, 128))
                    hp["astseq"]["rateDropout"] = trial.suggest_uniform('rateDropout', 0.0, 0.3)
                if(0 < len(dataset4Train.samples[0]["x"]["codemetrics"])):
                    hp["codemetrics"] = {}
                if(0 < len(dataset4Train.samples[0]["x"]["commitgraph"]["nodes"])):
                    hp["commitgraph"] = {}
                if(0 < len(dataset4Train.samples[0]["x"]["commitseq"])):
                    hp["commitseq"] = {}
                    hp["astseq"]["numLayers"] = trial.suggest_int('numLayers', 1, 3)
                    hp["astseq"]["numParameters"] = int(trial.suggest_int('numParameters', 16, 128))
                    hp["astseq"]["rateDropout"] = trial.suggest_uniform('rateDropout', 0.0, 0.3)
                if(0 < len(dataset4Train.samples[0]["x"]["processmetrics"])):
                    hp["processmetrics"] = {}
                hp["optimizer"] = trial.suggest_categorical('optimizer', ['adam'])
                hp["lrAdam"] = trial.suggest_loguniform('lrAdam', 1e-6, 1e-3)
                hp["beta1Adam"] = trial.suggest_uniform('beta1Adam', 0.9, 1)
                hp["beta2Adam"] = trial.suggest_uniform('beta2Adam', 0.999, 1)
                hp["epsilonAdam"] = trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-8)
                hp["sizeBatch"] = trial.suggest_int("sizeBatch", 16, 128)
                dataloader={
                    "train": DataLoader(
                        dataset4Train,
                        batch_size = hp["sizeBatch"],
                        pin_memory=True,
                        collate_fn = dataset4Train.collate_fn_my
                    ),
                    "valid": DataLoader(
                        dataset4Test,
                        batch_size = hp["sizeBatch"],
                        pin_memory=True,
                        collate_fn= dataset4Test.collate_fn_my
                    )
                }

                # prepare model architecture
                model = self.defineNetwork(hp, dataset4Train)

                # prepare loss function
                lossFunction = nn.BCELoss()

                # prepare  optimizer
                optimizer = self.getOptimizer(hp, model)

                # train!
                epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid = self.train_(dataloader, model, lossFunction, optimizer, numEpochs)
                self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, trial.number)
                trial.set_user_attr("numEpochs", epochBestValid+1)
                
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
            scoreAverage = scoreAverage / len(datasets_Train_Valid)
            #全体のログをloggerで出力
            with open(Result4BugPrediction.getPathLogSearchHyperParameter(), mode='a') as f:
                f.write(str(score)+","+str(trial.datetime_start)+","+str(trial.params)+'\n')
            return scoreAverage
        study = optuna.create_study()
        hpDefault = {}
        hpDefault["numLayers"] = 2
        hpDefault["numParameters"] = 128
        hpDefault["rateDropout"] = 0
        hpDefault["optimizer"] = "adam"
        hpDefault["lrAdam"] = 1e-3
        hpDefault["beta1Adam"] = 0.9
        hpDefault["beta2Adam"] = 0.999
        hpDefault["epsilonAdam"] = 1e-8
        hpDefault["sizeBatch"] = 32
        study.enqueue_trial(hpDefault)
        study.optimize(objectiveFunction, timeout=self.period4HyperParameterSearch)
        #save the hyperparameter that seems to be the best.
        #self.plotGraphHyperParameterSearch(study.get_trials())
        with open(Result4BugPrediction.getPathHyperParameter(), mode='w') as file:
            json.dump(dict(study.best_params.items()^study.best_trial.user_attrs.items()), file, indent=4)
        return Result4BugPrediction.getPathHyperParameter()

    def searchParameter(self, dataset4SearchParameter):
        with open(Result4BugPrediction.getPathHyperParameter(), mode='r') as file:
            hp = json.load(file)

        # prepare dataset
        dataset4Train = self.Dataset([list(i) for i in zip(*dataset4SearchParameter["train"])][1:])
        dataset4Test = self.Dataset([list(i) for i in zip(*dataset4SearchParameter["valid"])][1:])
        dataloader={
            "train": DataLoader(
                dataset4Train,
                batch_size = hp["sizeBatch"],
                pin_memory=True,
                collate_fn = self.collate_fn_my
            ),
            "valid": DataLoader(
                dataset4Test,
                batch_size = hp["sizeBatch"],
                pin_memory=True,
                collate_fn = self.collate_fn_my
            )
        }

        # prepare model architecture
        model = self.defineNetwork(hp, dataset4Train.getNumFeatures())

        # prepare loss function
        lossFunction = nn.BCELoss()

        # prepare  optimizer
        optimizer = self.getOptimizer(hp, model)

        # prepare model parameters
        _, lossesTrain, lossesValid, accsTrain, accsValid = self.train_(dataloader, model, lossFunction, optimizer, hp["numEpochs"])
        self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, 10000)

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
            "valid": DataLoader(
                dataset4Test,
                batch_size = hp["sizeBatch"],
                pin_memory=True,
                collate_fn = self.collate_fn_my
            )
        }

        # prepare model architecture
        model = self.defineNetwork(hp, dataset4Test.getNumFeatures())

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