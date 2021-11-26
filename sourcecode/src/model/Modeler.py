from src.config.config import config
from src.data.Dataset import Dataset
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
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from src.log.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.pathLog)

class Modeler(nn.Module):
    def __init__(self):
        super().__init__()
        self.trials4HyperParameterSearch = config.trials4HyperParameterSearch
        self.period4HyperParameterSearch = config.period4HyperParameterSearch
        self.epochsEarlyStopping = config.epochs4EarlyStopping
        self.device = config.device or "cuda:0"
        torch.backends.cudnn.enabled = True
        self.forward = None
    def defineNetwork(self, hp, dataset):
        print(hp)
        numOfFeatures = 0
        self.componentsNetwork = nn.ModuleDict(
            {
                "ast":nn.ModuleDict(),
                "astseq":nn.ModuleDict(),
                "codemetrics":nn.ModuleDict(),
                "commitgraph":nn.ModuleDict(),
                "commitseq": nn.ModuleDict(),
                "processmetrics":nn.ModuleDict(),
                "metrics": nn.ModuleDict()
            }
        )
        if(config.checkASTExists()):
            pass
        if(config.checkASTSeqExists()):
            isBidirectional = True
            self.componentsNetwork["astseq"]["LSTM"] = nn.LSTM(
                    input_size = len(dataset.astseqs[0][0]),
                    hidden_size = hp["astseq_hiddenSize"],
                    num_layers = hp["astseq_numOfLayers"],
                    batch_first = True,
                    dropout = hp["astseq_rateDropout"],
                    bidirectional = isBidirectional
                )
            if(isBidirectional == True):
                numOfFeatures_astseq = hp["astseq_hiddenSize"]*2
            else:
                numOfFeatures_astseq = hp["astseq_hiddenSize"]
            numOfFeatures += numOfFeatures_astseq
        if(config.checkCommitGraphExists()):
            pass
        if(config.checkCommitSeqExists()):
            isBidirectional = True
            self.componentsNetwork["commitseq"]["LSTM"] = nn.LSTM(
                    input_size = len(dataset.commitseqs[0][0]),
                    hidden_size = hp["commitseq_hiddenSize"],
                    num_layers = hp["commitseq_numOfLayers"],
                    batch_first = True,
                    dropout = hp["commitseq_rateDropout"],
                    bidirectional = isBidirectional
                )
            if(isBidirectional == True):
                numOfFeatures_commitseq = hp["commitseq_hiddenSize"]*2
            else:
                numOfFeatures_commitseq = hp["commitseq_hiddenSize"]
            numOfFeatures += numOfFeatures_commitseq
        if(config.checkCodeMetricsExists() or config.checkProcessMetricsExists()):
            numOfFeaturesMetrics = len(dataset.codemetricss[0])+len(dataset.processmetricss[0])
            for i in range(hp["metrics_numOfLayers"]):
                if( i == 0 ):
                    in_features = numOfFeaturesMetrics
                else:
                    in_features = hp["metrics_numOfOutput"]
                self.componentsNetwork["codemetrics"]["linear"+str(i)] = nn.Linear(
                    in_features = in_features,
                    out_features = hp["metrics_numOfOutput"]
                )
            numOfFeatures += hp["metrics_numOfOutput"]
        self.componentsNetwork["features"] = nn.Linear(numOfFeatures, 1)
        def forward(ast, astseq, codemetrics, commitgraph, commitseq, processmetrics):
            features = []
            if(config.checkASTExists()):
                pass
            if(config.checkASTSeqExists()):
                _, (parametersBiLSTM, _) = self.componentsNetwork["astseq"]["LSTM"](astseq)
                parametersBiLSTM = torch.cat(torch.split(parametersBiLSTM[(hp["astseq_numOfLayers"]-1)*2:], 1), dim=2)
                featuresFromASTSeq = parametersBiLSTM.squeeze()
                features.append(featuresFromASTSeq)
            if(config.checkCommitGraphExists()):
                pass
            if(config.checkCommitSeqExists()):
                _, (parametersBiLSTM, __) = self.componentsNetwork["commitseq"]["LSTM"](commitseq)
                parametersBiLSTM = torch.cat(torch.split(parametersBiLSTM[(hp["commitseq_numOfLayers"]-1)*2:], 1), dim=2)
                featuresFromCommitSeq = parametersBiLSTM.squeeze()
                features.append(featuresFromCommitSeq)
            if(config.checkCodeMetricsExists() and config.checkProcessMetricsExists()):
                featuresFromMetrics = torch.cat([codemetrics, processmetrics], dim=1)
                for i in range(hp["metrics_numOfLayers"]):
                    featuresFromMetrics = self.componentsNetwork["codemetrics"]["linear"+str(i)](featuresFromMetrics)#todo codemetricsだけ？
                features.append(featuresFromMetrics)
            elif(config.checkProcessMetricsExists()):
                featuresFromMetrics = processmetrics
                for i in range(hp["metrics_numOfLayers"]):
                    featuresFromMetrics = self.componentsNetwork["codemetrics"]["linear"+str(i)](featuresFromMetrics)
                features.append(featuresFromMetrics)
            features = torch.cat(features, dim = 1)
            y = self.componentsNetwork["features"](features)
            return y
        self.forward = forward
        model = self.to(self.device)
        summary(model)
        return model
    def defineOptimizer(self, hp, model):
        nameOptimizer = hp["optimizer"]
        if nameOptimizer == 'adam':
            lrAdam = hp["lrAdam"]
            beta_1Adam = hp["beta1Adam"]
            beta_2Adam = hp["beta2Adam"]
            epsilonAdam = hp["epsilonAdam"]
            optimizer = torch.optim.Adam(model.parameters(), lr=lrAdam, betas=(beta_1Adam,beta_2Adam), eps=epsilonAdam)
        return optimizer
    def searchParameter(self, dataLoader, model, lossFunction, optimizer, numEpochs, isEarlyStopping):
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
                    for asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss, ys in dataLoader[phase]:
                        if(config.checkASTExists()):
                            asts = asts.to(self.device)
                        if(config.checkASTSeqExists()):
                            astseqs = astseqs.to(self.device)
                        if(config.checkCommitGraphExists()):
                            commitgraphs = commitgraphs.to(self.device)
                        if(config.checkCommitSeqExists()):
                            commitseqs = commitseqs.to(self.device)
                        if(config.checkCodeMetricsExists()):
                            codemetricss = codemetricss.to(self.device)
                        if(config.checkProcessMetricsExists()):
                            processmetricss = processmetricss.to(self.device)
                        ys = ys.to(self.device)
                        ysPredicted = model(asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss)
                        ysPredicted = ysPredicted.squeeze()
                        ys = ys.squeeze()
                        loss=lossFunction(ysPredicted, ys)

                        if phase=="train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        sig = nn.Sigmoid()
                        ysPredicted =  torch.round(sig(ysPredicted))
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
            if(isEarlyStopping and self.epochsEarlyStopping<epoch-epochBestValid):
                break
        return epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid
    def loadHyperparameter(self):
        if(config.hyperparameter):
            return config.hyperparameter
        elif(config.pathDatabaseOptuna):
            # optuunaデータベースから、最適なハイパーパラメータを読み出す
            study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDirOutput + "/optuna.db", load_if_exists=True)
            return dict(study.best_params.items()^study.best_trial.user_attrs.items())
        else:
            raise Exception()
    def plotGraphTraining(self, lossesTrain, lossesValid, accTrain, accValid, title):
        epochs = range(len(lossesTrain))

        fig = plt.figure()
        plt.ylim(0, 2)
        plt.plot(epochs, lossesTrain, linestyle="-", color='b', label = 'lossTrain')
        plt.plot(epochs, accTrain, linestyle="-", color='r', label = 'accTrain')
        plt.plot(epochs, lossesValid, linestyle=":", color='b' , label= 'lossVal')
        plt.plot(epochs, accValid, linestyle=":", color='r' , label= 'accVal')
        plt.title(title)
        plt.legend()

        pathGraph = os.path.join(config.pathDirOutput, title + '.png')
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

        pathGraph = os.path.join(config.pathDirOutput, "hyperParameterSearch" + '.png')
        fig.savefig(pathGraph)
        plt.clf()
        plt.close()
    def searchHyperParameter(self, datasets_Train_Valid):
        logger.info("search hyper parameter")
        def objectiveFunction(trial):
            logger.info("trial " + str(trial.number))
            numEpochs = 10000
            lossesCrossValidation=0
            for index4CrossValidation in range(len(datasets_Train_Valid)):
                # prepare dataset
                dataset4Train = datasets_Train_Valid[index4CrossValidation]["train"]
                dataset4Valid = datasets_Train_Valid[index4CrossValidation]["valid"]
                hp = {}
                if(config.checkASTExists()):
                    pass
                if(config.checkASTSeqExists()):
                    hp["astseq_numOfLayers"] = trial.suggest_int('astseq_numOfLayers', 1, 3)
                    hp["astseq_hiddenSize"] = trial.suggest_int('astseq_hiddenSize', 16, 256)
                    hp["astseq_rateDropout"] = trial.suggest_uniform('astseq_rateDropout', 0.0, 0.0)#trial.suggest_uniform('astseq_rateDropout', 0.0, 0.3)
                if(config.checkCommitGraphExists()):
                    pass
                if(config.checkCommitSeqExists()):
                    hp["commitseq_numOfLayers"] = trial.suggest_int('commitseq_numOfLayers', 1, 3)
                    hp["commitseq_hiddenSize"] = trial.suggest_int('commitseq_hiddenSize', 16, 256)
                    hp["commitseq_rateDropout"] = trial.suggest_uniform('commitseq_rateDropout', 0.0, 0.0)#trial.suggest_uniform('rateDropout', 0.0, 0.3)
                if(config.checkCodeMetricsExists() or config.checkProcessMetricsExists()):
                    hp["metrics_numOfLayers"] = trial.suggest_int('metrics_numOfLayers', 1, 3)
                    hp["metrics_numOfOutput"] = trial.suggest_int('metrics_numOfOutput', 16, 128)
                else:
                    if(config.checkCodeMetricsExists()):
                        hp["codemetrics"] = {}
                    if(config.checkProcessMetricsExists()):
                        hp["processmetrics"] = {}
                hp["optimizer"] = trial.suggest_categorical('optimizer', ['adam'])
                hp["lrAdam"] = trial.suggest_loguniform('lrAdam', 1e-6, 1e-4)
                hp["beta1Adam"] = trial.suggest_uniform('beta1Adam', 0.9, 0.9)#trial.suggest_uniform('beta1Adam', 0.9, 1)
                hp["beta2Adam"] = trial.suggest_uniform('beta2Adam', 0.999, 0.999)#trial.suggest_uniform('beta2Adam', 0.999, 1)
                hp["epsilonAdam"] = trial.suggest_loguniform('epsilonAdam', 1e-8, 1e-8) #trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-8)
                hp["sizeBatch"] = trial.suggest_int('sizeBatch', 128, 128) #trial.suggest_int('sizeBatch', 16, 128)
                dataloader={
                    "train": DataLoader(
                        dataset4Train,
                        batch_size = hp['sizeBatch'],
                        pin_memory=False,
                        collate_fn = dataset4Train.collate_fn
                    ),
                    "valid": DataLoader(
                        dataset4Valid,
                        batch_size = hp['sizeBatch'],
                        pin_memory=False,
                        collate_fn= dataset4Valid.collate_fn
                    )
                }

                # prepare model architecture
                model = self.defineNetwork(hp, dataset4Train)

                # prepare loss function
                lossFunction = nn.BCEWithLogitsLoss()

                # prepare  optimizer
                optimizer = self.defineOptimizer(hp, model)

                # train!
                epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid = self.searchParameter(dataloader, model, lossFunction, optimizer, numEpochs, isEarlyStopping=True)
                self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, "graphTrainToValid_" + str(trial.number) + "_" + str(index4CrossValidation))
                trial.set_user_attr("numEpochs", epochBestValid+1)

                # 1エポックだけ偶然高い精度が出たような場合を弾くために、前後のepochで平均を取る。
                lossValMin = min(lossesValid)
                indexValMin = lossesValid.index(lossValMin)
                indexLast = len(lossesValid)-1
                index4Forward = indexValMin+4 if indexValMin+4 < indexLast else indexLast
                loss=0
                for i in range(5):
                    loss += lossesValid[index4Forward-i]
                loss = loss / 5
                lossesCrossValidation += loss
            value2BeOptimized = lossesCrossValidation / len(datasets_Train_Valid)
            return value2BeOptimized
        study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDirOutput + "/optuna.db", load_if_exists=True)
        if(len(study.get_trials())==0):
            if(config.checkCommitSeqExists() & config.checkASTSeqExists() & config.checkCodeMetricsExists() & config.checkProcessMetricsExists()):
                hp_default = {
                    "astseq_numOfLayers": 2,
                    "astseq_hiddenSize": 128,
                    "astseq_rateDropout": 0.0,
                    "commitseq_numOfLayers": 2,
                    "commitseq_hiddenSize": 128,
                    "commitseq_rateDropout": 0.0,
                    "metrics_numOfLayers": 2,
                    "metrics_numOfOutput": 64,
                    "sizeBatch": 128,
                    "optimizer": "adam",
                    "lrAdam": 1e-05,
                    "beta1Adam": 0.9,
                    "beta2Adam": 0.999,
                    "epsilonAdam": 1e-08
                }
            elif(config.checkCommitSeqExists() & config.checkASTSeqExists()):
                hp_default = {
                    "astseq_numOfLayers": 2,
                    "astseq_hiddenSize": 128,
                    "astseq_rateDropout": 0.0,
                    "commitseq_numOfLayers": 2,
                    "commitseq_hiddenSize": 128,
                    "commitseq_rateDropout": 0.0,
                    "sizeBatch": 128,
                    "optimizer": "adam",
                    "lrAdam": 1e-05,
                    "beta1Adam": 0.9,
                    "beta2Adam": 0.999,
                    "epsilonAdam": 1e-08
                }
            elif(config.checkCodeMetricsExists() & config.checkProcessMetricsExists()):
                hp_default = {
                    "metrics_numOfLayers": 2,
                    "metrics_numOfOutput": 64,
                    "sizeBatch": 128,
                    "optimizer": "adam",
                    "lrAdam": 1e-05,
                    "beta1Adam": 0.9,
                    "beta2Adam": 0.999,
                    "epsilonAdam": 1e-08
                }
            elif(config.checkCommitSeqExists()):
                hp_default = {
                    "commitseq_numOfLayers": 2,
                    "commitseq_hiddenSize": 128,
                    "commitseq_rateDropout": 0.0,
                    "sizeBatch": 128,
                    "optimizer": "adam",
                    "lrAdam": 1e-05,
                    "beta1Adam": 0.9,
                    "beta2Adam": 0.999,
                    "epsilonAdam": 1e-08
                }
            elif(config.checkProcessMetricsExists()):
                hp_default = {
                    "metrics_numOfLayers": 2,
                    "metrics_numOfOutput": 64,
                    "sizeBatch": 128,
                    "optimizer": "adam",
                    "lrAdam": 1e-05,
                    "beta1Adam": 0.9,
                    "beta2Adam": 0.999,
                    "epsilonAdam": 1e-08
                }
            study.enqueue_trial(hp_default)
        study.optimize(objectiveFunction, timeout=config.period4HyperParameterSearch)
        #save the hyperparameter that seems to be the best.
        self.plotGraphHyperParameterSearch(study.get_trials())
        config.pathHyperParameters = os.path.join(config.pathDirOutput, "hyperParameter.json")
        with open(config.pathHyperParameters, mode='w') as file:
            json.dump(dict(study.best_params.items()^study.best_trial.user_attrs.items()), file, indent=4)
    def buildModel(self, datasets_Train_Test):
        logger.info("search parameter")
        with open(config.pathHyperParameters, mode='r') as file:
            hp = json.load(file)

        # prepare dataset
        dataset4Train = datasets_Train_Test["train"]
        dataset4Test = datasets_Train_Test["test"]
        dataloader={
            "train": DataLoader(
                dataset4Train,
                batch_size = hp["sizeBatch"],
                pin_memory=True,
                collate_fn = dataset4Train.collate_fn
            ),
            "valid": DataLoader(
                dataset4Test,
                batch_size = hp["sizeBatch"],
                pin_memory=True,
                collate_fn = dataset4Test.collate_fn
            )
        }

        # prepare network architecture
        model = self.defineNetwork(hp, dataset4Train)

        # prepare loss function
        lossFunction = nn.BCEWithLogitsLoss()

        # prepare  optimizer
        optimizer = self.defineOptimizer(hp, model)

        # prepare model parameters
        _, lossesTrain, lossesValid, accsTrain, accsValid = self.searchParameter(dataloader, model, lossFunction, optimizer, hp["numEpochs"], isEarlyStopping=False)
        self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, "graphTrainToTest")

        config.pathParameters = os.path.join(config.pathDirOutput, "parameters")
        torch.save(model.state_dict(), config.pathParameters)
    def testModel(self, datasetTest):
        logger.info("test parameter")

        # get hyperparameter
        hp = self.loadHyperparameter()

        # get dataset
        dataset4Test = datasetTest
        dataloader={
            "test": DataLoader(
                dataset4Test,
                batch_size = hp["sizeBatch"],
                pin_memory=False,
                collate_fn = dataset4Test.collate_fn
            )
        }

        # define network architecture
        model = self.defineNetwork(hp, dataset4Test)

        # build model
        paramaters = torch.load(config.pathParameters)
        model.load_state_dict(paramaters)
        model = model.eval()

        # predict ys
        yPredicted = []
        yTest = []
        for asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss, ys in dataloader['test']:
            if(config.checkASTExists()):
                asts = asts.to(self.device)
            if(config.checkASTSeqExists()):
                astseqs = astseqs.to(self.device)
            if(config.checkCommitGraphExists()):
                commitgraphs = commitgraphs.to(self.device)
            if(config.checkCommitSeqExists()):
                commitseqs = commitseqs.to(self.device)
            if(config.checkCodeMetricsExists()):
                codemetricss = codemetricss.to(self.device)
            if(config.checkProcessMetricsExists()):
                processmetricss = processmetricss.to(self.device)
            with torch.no_grad():
                output = model(asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss)
                sig = nn.Sigmoid()
                output = sig(output)
                yPredicted += [l for l in output.to("cpu").squeeze().tolist()]
                yTest += [l for l in ys.to("cpu").squeeze().tolist()]

        # output prediction result
        IDRecord = [list(i) for i in zip(*datasetTest)][0]
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