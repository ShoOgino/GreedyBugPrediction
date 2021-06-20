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
from torch.utils.data import DataLoader
from torchsummary import summary 
from torchvision import transforms
from tqdm import tqdm

class DNN4BugPrediction_Pytorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.trials4HyperParameterSearch = 100
        self.isCrossValidation = True
        self.device = "cuda:0"

    def forward(self, x):
        logits = self.architecture(x)
        return logits

    def setTrials4HyperParameterSearch(self, trials4HyperParameterSearch):
        self.trials4HyperParameterSearch = trials4HyperParameterSearch

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
        class Dataset4SearchHyperParameter(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
                self.dataset[0] = torch.tensor(dataset[0]).float()
                print(self.dataset[0].shape)
                self.dataset[1] = torch.tensor(dataset[1]).float()
                print(self.dataset[1].shape)
            def __len__(self):
                return len(self.dataset[0])
            def __getitem__(self, index):
                return self.dataset[1][index], self.dataset[0][index]
        def objectiveFunction(trial):
            def chooseModel(trial, numFeatures):
                sizeOutput = 1
                numLayers = trial.suggest_int('numlayers', 1, 3)
                self.architecture = nn.Sequential()
                numInput = numFeatures
                for i in range(numLayers):
                    numOutput = int(trial.suggest_int('numUnits{}'.format(i), 16, 1024))
                    rateDropout = trial.suggest_uniform('rateDropout{}'.format(i), 0.0, 0.5)
                    activation = trial.suggest_categorical('activation{}'.format(i), ['hard_sigmoid', 'relu', 'sigmoid', 'softplus','softsign', 'tanh'])
                    self.architecture.add_module("layer"+str(i), nn.Linear(numInput, numOutput))
                    if(activation=='hard_sigmoid'): self.architecture.add_module("activation"+str(i), nn.Hardsigmoid())
                    elif(activation=='relu'): self.architecture.add_module("activation"+str(i), nn.ReLU())
                    elif(activation=='sigmoid'): self.architecture.add_module("activation"+str(i), nn.Sigmoid())
                    elif(activation=='softplus'):self.architecture.add_module("activation"+str(i), nn.Softplus())
                    elif(activation=='softsign'): self.architecture.add_module("activation"+str(i), nn.Softsign())
                    elif(activation=='tanh'): self.architecture.add_module("activation"+str(i), nn.Tanh())
                    self.architecture.add_module("dropout"+str(i), nn.Dropout(p=rateDropout))
                    numInput = numOutput
                self.architecture.add_module("output", nn.Linear(numInput, sizeOutput))
                self.architecture.add_module("activationOutput", nn.Sigmoid())
            def chooseOptimizer(trial, model):
                nameOptimizer = trial.suggest_categorical('optimizer', ['adam'])
                if nameOptimizer == 'adam':
                    lrAdam = trial.suggest_loguniform('lrAdam', 1e-5, 1e-2)
                    beta_1Adam = trial.suggest_uniform('beta1Adam', 0.9, 1)
                    beta_2Adam = trial.suggest_uniform('beta2Adam', 0.999, 1)
                    epsilonAdam = trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-5)
                    self.optimizer = torch.optim.Adam(model.parameters(), lr=lrAdam, betas=(beta_1Adam,beta_2Adam), eps=epsilonAdam)
            numEpochs, sizeBatch = 10000, trial.suggest_int("sizeBatch", 16, 128)
            for index4CrossValidation in range(len(arrayOfD4TAndD4V)):
                dataset4Train = Dataset4SearchHyperParameter([list(i) for i in zip(*arrayOfD4TAndD4V[index4CrossValidation]["training"])][1:])
                dataset4Test = Dataset4SearchHyperParameter([list(i) for i in zip(*arrayOfD4TAndD4V[index4CrossValidation]["validation"])][1:])
                dataloader={
                    "train": DataLoader(dataset4Train, batch_size = sizeBatch, pin_memory=True),
                    "valid": DataLoader(dataset4Test, batch_size = sizeBatch, pin_memory=True)
                }
                scoreAverage=0
                numFeatures = len(dataset4Train.__getitem__(0)[0])
                print(numFeatures)
                chooseModel(trial, numFeatures)
                model = self.to(self.device)
                summary(model, input_size=(1 , 24))
                chooseOptimizer(trial, model)
                loss_fn = nn.BCELoss()

                lossesTrain = []
                lossesValid = []
                accsTrain = []
                accsValid = []
                lossValidBest = 10000
                lastEpochBestValid = 0
                for epoch in range(numEpochs):
                    for phase in ["train","valid"]:
                        if phase=="train":
                            model.train()
                        elif phase=="valid":
                            model.eval()
                        loss_sum=0
                        corrects=0
                        total=0
                        with tqdm(total=len(dataloader[phase]),unit="batch") as pbar:
                            pbar.set_description(f"Epoch[{epoch}/{numEpochs}]({phase})")
                            for xs, ys in dataloader[phase]:   
                                xs, ys = xs.to(self.device), ys.to(self.device)
                                ysPredicted=model(xs)
                                ys = ys.unsqueeze(1)
                                loss=loss_fn(ysPredicted, ys)

                                if phase=="train":
                                    self.optimizer.zero_grad()
                                    loss.backward()
                                    self.optimizer.step()

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
                                lastEpochBestValid = epoch
                    if(100<epoch-lastEpochBestValid):
                        break
                self.plotLearningCurve(lossesTrain, lossesValid, accsTrain, accsValid, trial.number)
                trial.set_user_attr("numEpochs", lastEpochBestValid)
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

    def searchParameter(self, dataset4Training):
        xTrain = np.array([list(i) for i in zip(*dataset4Training)][2])
        yTrain = np.array([list(i) for i in zip(*dataset4Training)][1])
        with open(Result4BugPrediction.getPathHyperParameter(), mode='r') as file:
            hp = json.load(file)
        sizeOutput = 1
        model = Sequential()
        for i in range(int(hp["numlayers"])):
            model.add(Dense(hp["numUnits"+str(i)], activation=hp["activation"+str(i)]))
            model.add(Dropout(hp["rateDropout"+str(i)]))
            model.add(Dense(sizeOutput,activation='sigmoid'))
        verbose, numEpochs, sizeBatch = 1, hp["numEpochs"], hp["sizeBatch"]
        numFeatures, sizeOutput = xTrain.shape[1], 1
        model.build((None,numFeatures))
        if hp["optimizer"] == 'adam':
            lrAdam = hp["lrAdam"]
            beta_1Adam = hp["beta1Adam"]
            beta_2Adam = hp["beta2Adam"]
            epsilonAdam = hp["epsilonAdam"]
            opt = optimizers.Adam(learning_rate=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        xTrain = np.array([list(i) for i in zip(*dataset4Training)][2])
        yTrain = np.array([list(i) for i in zip(*dataset4Training)][1])
        model.fit(xTrain, yTrain, epochs=numEpochs, batch_size=sizeBatch, verbose=verbose)

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