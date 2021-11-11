import argparse
import glob
import json
import csv
import os


def main(pathDir):
    with open("test.csv", "w", newline="") as fcsv:
        reportCSV = csv.writer(fcsv)
        reportCSV.writerow(["", "accuracy", "precision", "recall", "f-measure", "AUC"])
        pathsProjectDir = glob.glob(pathDir+"/*")
        for pathProjectDir in pathsProjectDir:
            pathsResultDir = glob.glob(pathProjectDir+"/*/")
            for pathResultDir in pathsResultDir:
                with open(pathResultDir+"/report.json", "r") as finput:
                    reportJSON = json.load(finput)
                    reportCSV.writerow([os.path.basename(pathProjectDir)+os. path.basename(os.path.dirname(pathResultDir)), reportJSON["accuracy"], reportJSON["1.0"]["precision"], reportJSON["1.0"]["recall"], reportJSON["1.0"]["f1-score"], reportJSON["AUC"]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Convert a git log output to json.
                                                 """)
    parser.add_argument('--pathDir', type=str)

    args = parser.parse_args()
    pathDir = args.pathDir
    main(pathDir)