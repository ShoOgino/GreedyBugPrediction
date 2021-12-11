from src.config.config import config
from src.data.Dataset import Dataset
from src.log.wrapperLogger import wrapperLogger
logger = wrapperLogger.setup_logger(__name__, config.pathLog)
import torch
import torch.nn.functional as F
import torch.nn as nn
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
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
        numOfFeatures = 0
        self.componentsNetwork = nn.ModuleDict(
            {
                "ast":nn.ModuleDict(),
                "astseq":nn.ModuleDict(),
                "codemetrics":nn.ModuleDict(),
                "commitgraph":nn.ModuleDict(),
                "commitseq": nn.ModuleDict(),
                "processmetrics":nn.ModuleDict(),
                "features": nn.ModuleDict()
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
        if(config.checkCodeMetricsExists()):
            numOfFeaturesMetrics = len(dataset.codemetricss[0])
            for i in range(hp["codemetrics_numOfLayers"]):
                if( i == 0 ):
                    in_features = numOfFeaturesMetrics
                else:
                    in_features = hp["codemetrics_numOfOutput"]
                self.componentsNetwork["codemetrics"]["linear"+str(i)] = nn.Linear(
                    in_features = in_features,
                    out_features = hp["codemetrics_numOfOutput"]
                )
            numOfFeatures += hp["codemetrics_numOfOutput"]
        if(config.checkProcessMetricsExists()):
            numOfFeaturesMetrics = len(dataset.processmetricss[0])
            for i in range(hp["processmetrics_numOfLayers"]):
                if( i == 0 ):
                    in_features = numOfFeaturesMetrics
                else:
                    in_features = hp["processmetrics_numOfOutput"]
                self.componentsNetwork["processmetrics"]["linear"+str(i)] = nn.Linear(
                    in_features = in_features,
                    out_features = hp["processmetrics_numOfOutput"]
                )
            numOfFeatures += hp["processmetrics_numOfOutput"]
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
            if(config.checkCodeMetricsExists()):
                featuresFromMetrics = codemetrics
                for i in range(hp["metrics_numOfLayers"]):
                    featuresFromMetrics = self.componentsNetwork["codemetrics"]["linear"+str(i)](featuresFromMetrics)
                features.append(featuresFromMetrics)
            if(config.checkProcessMetricsExists()):
                featuresFromMetrics = processmetrics
                for i in range(hp["processmetrics_numOfLayers"]):
                    featuresFromMetrics = self.componentsNetwork["processmetrics"]["linear"+str(i)](featuresFromMetrics)
                features.append(featuresFromMetrics)
            features = torch.cat(features, dim = 1)
            y = self.componentsNetwork["features"](features)
            return y
        self.forward = forward
        model = self.to(self.device)
        return model
    def initParameter(self):
        if(config.checkASTExists()):
            pass
        if(config.checkASTSeqExists()):
            for name, param in self.componentsNetwork["astseq"]["LSTM"].named_parameters():
                if 'bias' in name:
                   nn.init.constant(param, 0.0)
                elif 'weight' in name:
                   nn.init.xavier_normal(param)
        if(config.checkCommitGraphExists()):
            pass
        if(config.checkCommitSeqExists()):
            for name, param in self.componentsNetwork["commitseq"]["LSTM"].named_parameters():
                if 'bias' in name:
                   nn.init.constant(param, 0.0)
                elif 'weight' in name:
                   nn.init.xavier_normal(param)
        if(config.checkCodeMetricsExists()):
            for layer in self.componentsNetwork["codemetrics"]:
                nn.init.normal_(self.componentsNetwork["codemetrics"][layer].weight, 0.0, 1.0)
        if(config.checkProcessMetricsExists()):
            for layer in self.componentsNetwork["processmetrics"]:
                nn.init.normal_(self.componentsNetwork["processmetrics"][layer].weight, 0.0, 1.0)
        nn.init.normal_(self.componentsNetwork["features"].weight, 0.0, 1.0)
    def defineOptimizer(self, hp, model):
        nameOptimizer = hp["optimizer"]
        if nameOptimizer == 'adam':
            lrAdam = hp["lrAdam"]
            beta_1Adam = hp["beta1Adam"]
            beta_2Adam = hp["beta2Adam"]
            epsilonAdam = hp["epsilonAdam"]
            optimizer = torch.optim.Adam(model.parameters(), lr=lrAdam, betas=(beta_1Adam,beta_2Adam), eps=epsilonAdam)
        return optimizer
    def searchParameter(self, *, dataLoader, model, lossFunction, optimizer, numEpochs, isEarlyStopping):
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
                    ysPredicted = ysPredicted.squeeze()#もしysが1つしかなかったら、ベクトルじゃなくてスカラーに鳴ってしまう
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
                    #loss関数で通してでてきたlossはCrossEntropyLossのreduction="mean"なので平均
                    #batch sizeをかけることで、batch全体での合計を今までのloss_sumに足し合わせる
                    loss_sum += float(loss) * ys.size(0)
                if(phase == "train"):
                    lossesTrain.append(loss_sum/total)
                    accsTrain.append(corrects/total)
                if(phase == "valid"):
                    lossesValid.append(loss_sum/total)
                    accsValid.append(corrects/total)
                    if(loss_sum < lossValidBest):
                        lossValidBest = loss_sum
                        epochBestValid = epoch
            if(isEarlyStopping and self.epochsEarlyStopping<epoch-epochBestValid):
                break
        return epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid
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
    def plotGraphTraining(self, lossesTrain, lossesValid, accTrain, accValid, title):
        # todo: lossが最小値のところに縦線・横線を引く
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
        # todo: lossが最小値のところに縦線・横線を引く
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
        def objectiveFunction(trial):
            logger.info("trial " + str(trial.number) + "started")
            listLossesValid=[]
            listEpochs=[]
            # prepare hyperparameter
            hp = {
                "optimizer": trial.suggest_categorical('optimizer', ['adam']),
                "lrAdam": trial.suggest_loguniform('lrAdam', 1e-6, 1e-4),
                "beta1Adam": trial.suggest_uniform('beta1Adam', 0.9, 0.9), #trial.suggest_uniform('beta1Adam', 0.9, 1)
                "beta2Adam": trial.suggest_uniform('beta2Adam', 0.999, 0.999), #trial.suggest_uniform('beta2Adam', 0.999, 1)
                "epsilonAdam": trial.suggest_loguniform('epsilonAdam', 1e-8, 1e-8), #trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-8)
                "sizeBatch": trial.suggest_int('sizeBatch', len(datasets_Train_Valid[0]["train"]) // 100, len(datasets_Train_Valid[0]["train"]) // 100)
            }
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
            if(config.checkCodeMetricsExists()):
                hp["codemetrics_numOfLayers"] = trial.suggest_int('codemetrics_numOfLayers', 1, 3)
                hp["codemetrics_numOfOutput"] = trial.suggest_int('codemetrics_numOfOutput', 16, 128)
            if(config.checkProcessMetricsExists()):
                hp["processmetrics_numOfLayers"] = trial.suggest_int('processmetrics_numOfLayers', 1, 3)
                hp["processmetrics_numOfOutput"] = trial.suggest_int('processmetrics_numOfOutput', 16, 128)
            # prepare model architecture
            model = self.defineNetwork(hp, datasets_Train_Valid[0]["train"])
            # prepare loss function
            lossFunction = nn.BCEWithLogitsLoss()
            # prepare  optimizer
            optimizer = self.defineOptimizer(hp, model)
            for index4CrossValidation in range(len(datasets_Train_Valid)):
                logger.info("cross validation " + str(index4CrossValidation+1) + "/" + str(len(datasets_Train_Valid)))
                # prepare dataset
                dataset4Train = datasets_Train_Valid[index4CrossValidation]["train"]
                dataset4Valid = datasets_Train_Valid[index4CrossValidation]["valid"]
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
                # build model
                self.initParameter()
                epochBestValid, lossesTrain, lossesValid, accsTrain, accsValid = self.searchParameter(
                    dataLoader = dataloader,
                    model = model,
                    lossFunction = lossFunction,
                    optimizer = optimizer,
                    numEpochs = 10000,
                    isEarlyStopping=config.epochs4EarlyStopping
                )
                listLossesValid.append(lossesValid)
                listEpochs.append(epochBestValid)
                self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, "graphTrainToValid_" + str(trial.number) + "_" + str(index4CrossValidation))
            # 最適なnumEpochsと、その時のハイパーパラメータの評価値を特定する。
            #sumOfSquaresBest = 10000
            #numEpochsBest = 10000
            #for epoch in range(min([len(item) for item in listLossesValid]) - 5):
            #    listLossValid = []
            #    # 1エポックだけ偶然高い精度が出たような場合を弾くために、前後のepochで平均を取る。
            #    for lossesValid in listLossesValid:
            #        temp = 0
            #        for i in range (5):
            #            temp += lossesValid[epoch+i]
            #        temp = temp/5
            #        listLossValid.append(temp)
            #    # 二乗の和
            #    sumOfSquares = 0
            #    for lossValid in listLossValid:
            #        sumOfSquares+=lossValid*lossValid
            #    if(sumOfSquares<sumOfSquaresBest):
            #        sumOfSquaresBest = sumOfSquares
            #        numEpochsBest = epoch
            numEpochsBest=sum(listEpochs)//len(listEpochs)
            trial.set_user_attr("numEpochs", numEpochsBest)
            sumOfSquare=0
            for i,numEpochs in enumerate(listEpochs):
                temp = sum(listLossesValid[i][numEpochs:numEpochs+5])/5
                sumOfSquare += temp * temp
            logger.info(
                "trial " + str(trial.number) + " end" + "\n" +
                "value: " + str(sumOfSquare) + "\n" +
                str(dict(**hp, **{"numEpochs":numEpochsBest})).replace("\'", "\"")
            )
            return sumOfSquare
        logger.info("hyperparameter search started")
        config.pathDatabaseOptuna = config.pathDatabaseOptuna or config.pathDirOutput + "/optuna.db"
        study = optuna.create_study(study_name="optuna", storage='sqlite:///'+config.pathDatabaseOptuna, load_if_exists=True)
        if(len(study.get_trials())==0):
            hp_default = {
                "sizeBatch": len(datasets_Train_Valid[0]["train"])//100,
                "optimizer": "adam",
                "lrAdam": 1e-05,
                "beta1Adam": 0.9,
                "beta2Adam": 0.999,
                "epsilonAdam": 1e-08
            }
            if(config.checkCommitSeqExists()):
                hp_default_commitseq ={
                    "commitseq_numOfLayers": 2,
                    "commitseq_hiddenSize": 128,
                    "commitseq_rateDropout": 0.0,
                }
                hp_default = dict(**hp_default, **hp_default_commitseq)
            if(config.checkASTSeqExists()):
                hp_default_ASTSeq ={
                    "astseq_numOfLayers": 2,
                    "astseq_hiddenSize": 128,
                    "astseq_rateDropout": 0.0,
                }
                hp_default = dict(**hp_default, **hp_default_ASTSeq)
            if(config.checkCodeMetricsExists()):
                hp_default_CodeMetrics ={
                    "codemetrics_numOfLayers": 2,
                    "codemetrics_numOfOutput": 64,
                }
                hp_default = dict(**hp_default, **hp_default_CodeMetrics)
            if(config.checkProcessMetricsExists()):
                hp_default_processmetrics = {
                    "processmetrics_numOfLayers": 2,
                    "processmetrics_numOfOutput": 64,
                }
                hp_default = dict(**hp_default, **hp_default_processmetrics)
            study.enqueue_trial(hp_default)
        study.optimize(objectiveFunction, timeout=config.period4HyperParameterSearch)
        #save the hyperparameter that seems to be the best.
        self.plotGraphHyperParameterSearch([v.value for v in study.trials])
    def buildModel(self, datasets_Train_Test):
        logger.info("build Model")

        hp = self.loadHyperparameter()

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
        _, lossesTrain, lossesValid, accsTrain, accsValid = self.searchParameter(
            dataLoader = dataloader,
            model = model,
            lossFunction = lossFunction,
            optimizer = optimizer,
            numEpochs = hp["numEpochs"],
            isEarlyStopping=False
        )
        self.plotGraphTraining(lossesTrain, lossesValid, accsTrain, accsValid, "graphTrainToTest")

        config.pathParameters = os.path.join(config.pathDirOutput, "parameters")
        torch.save(model.state_dict(), config.pathParameters)
    def testModel(self, datasetTest):
        logger.info("test model")

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