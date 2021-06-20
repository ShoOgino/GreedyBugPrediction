from os import path


class Result4BugPrediction:
    pathResult = ""
    pathLogSearchHyperParameter=""
    pathLogSearchParameter=""
    pathHyperParameter = ""
    pathParameter = ""
    def clear():
        Result4BugPrediction.pathResult=""
        Result4BugPrediction.pathLogSearchHyperParameter=""
        Result4BugPrediction.pathLogSearchParameter=""
        Result4BugPrediction.pathHyperParameter = ""
        Result4BugPrediction.pathParameter = ""
    def setPathResult(pathResult_):
        Result4BugPrediction.pathResult = pathResult_
    def getPathResult():
        return Result4BugPrediction.pathResult
    def setPathLogSearchHyperParameter(pathLogSearchHyperParameter):
        Result4BugPrediction.pathLogSearchHyperParameter = pathLogSearchHyperParameter
    def getPathLogSearchHyperParameter():
        if(Result4BugPrediction.pathLogSearchHyperParameter==""):
            return Result4BugPrediction.pathResult + "/logSearchHyperParameter.txt"
        else:
            return Result4BugPrediction.pathLogSearchHyperParameter
    def setPathLogHyperParameter(pathLogSearchParameter):
        Result4BugPrediction.pathLogSearchParameter = pathLogSearchParameter
    def getPathHyperParameter():
        if(Result4BugPrediction.pathHyperParameter==""):
            return Result4BugPrediction.pathResult + "/hyperparameter.json"
        else:
            return Result4BugPrediction.pathHyperParameter
    def setPathHyperParameter(pathHyperParameter):
        Result4BugPrediction.pathHyperParameter = pathHyperParameter
    def setPathParameter(pathParameter):
        Result4BugPrediction.pathParameter = pathParameter
    def getPathParameter(): 
        if(Result4BugPrediction.pathParameter==""):
            return Result4BugPrediction.pathResult + "/parameter"
        else:
            return Result4BugPrediction.pathParameter