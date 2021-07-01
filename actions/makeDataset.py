from transformers import AutoTokenizer, AutoModel
import torch
import glob
import os
import numpy as np 
import csv
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
# ファイルごとにトークン列のベクトルを作って保存。バグの有無も考える
bugdata ={}
records = []
pathProject = r"C:\Users\login\data\workspace\greedyBugPrediction\MLTool\datasets\egit"
pathRepositoryMethod = pathProject+"/repositoryMethod"
pathBugdataCSV = r"C:\Users\login\data\workspace\greedyBugPrediction\MLTool\datasets\egit\datasets\R5_r_test.csv"
with open(pathBugdataCSV, encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        bugdata[row[0]]=row[1]
for path in bugdata:
    pathMjava = pathRepositoryMethod + "/" + path
    print(pathMjava)
    if(os.path.exists(pathMjava)):
        with open(pathMjava, encoding="utf-8") as f:
            source = f.read()
            tokens = tokenizer.tokenize(source)
            print(tokens)
            seg = len(tokens)//510
            print("seg: "+str(seg))
            if(0<seg):
                vectors = np.empty((0,768), float)
                for indexOfSeg in range(seg):
                    if(len(tokens)<510*(indexOfSeg+1)):
                        end = len(tokens)-1
                    else:
                        end = 510*(indexOfSeg+1)
                    tokensSeg = tokens[510*indexOfSeg:end]
                    tokensSeg = [tokenizer.cls_token]+tokensSeg+[tokenizer.sep_token]
                    tokenIDs=tokenizer.convert_tokens_to_ids(tokensSeg)
                    context_embeddings_seg=model(torch.tensor(tokenIDs)[None,:])[0][0]
                    vectorNumpy = context_embeddings_seg[0].to('cpu').detach().numpy().copy()
                    vectors = np.append(vectors, [vectorNumpy], axis=0)
                average = np.average(vectors, axis = 0)
                records.append([path, bugdata[path], *(average.tolist())])
            else:
                tokens = [tokenizer.cls_token]+tokens+[tokenizer.sep_token]
                print(len(tokens))
                tokenIDs=tokenizer.convert_tokens_to_ids(tokens)
                print(len(tokenIDs))
                context_embeddings=model(torch.tensor(tokenIDs)[None,:])[0][0]
                records.append([path, bugdata[path], *(context_embeddings.detach().tolist())])

with open("records.csv", encoding="utf-8", mode="w") as f:
    writer = csv.writer(f)
    writer.writerows(records)

#token_embeddings=model.embeddings.word_embeddings(torch.tensor(tokens_ids))
#print(token_embeddings)
#print("length: "+ str(len(token_embeddings)))