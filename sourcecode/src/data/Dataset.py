from src.config.config import config
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.ids = []
        self.asts = []
        self.astseqs = []
        self.codemetricss = []
        self.commitgraphs = []
        self.commitseqs = []
        self.processmetricss = []
        self.ys = []
        for sample in samples:
            self.ids.append(sample["id"])
            self.asts.append(sample["x"]["ast"])
            self.astseqs.append(sample["x"]["astseq"])
            self.codemetricss.append(sample["x"]["codemetrics"])
            self.commitgraphs.append(sample["x"]["commitgraph"])
            self.commitseqs.append(sample["x"]["commitseq"])
            self.processmetricss.append(sample["x"]["processmetrics"])
            self.ys.append(sample["y"])
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        return self.ids[index], self.asts[index], self.astseqs[index], self.codemetricss[index], self.commitgraphs[index], self.commitseqs[index], self.processmetricss[index], self.ys[index]
    def getNumOfNegatives(self):
        return len([item for item in self.ys if item==0])
    def getNumOfPositives(self):
        return len([item for item in self.ys if item==1])
    def collate_fn(self, batch):
        ids, asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss, ys = list(zip(*batch))

        if(config.checkASTExists()):
            pass

        if(config.checkASTSeqExists()):
            astseqs = [torch.tensor(astseq).float() for astseq in astseqs]
            astseqsLength = torch.tensor([len(astseq) for astseq in astseqs])
            astseqsPadded = pad_sequence(astseqs, batch_first=True)
            astseqs = pack_padded_sequence(astseqsPadded, astseqsLength, batch_first=True, enforce_sorted=False)

        if(config.checkCodeMetricsExists()):
            codemetricss = torch.tensor(codemetricss).float()#[torch.tensor(codemetrics).float() for codemetrics in codemetricss]

        if(config.checkCommitGraphExists()):
            pass

        if(config.checkCommitSeqExists()):
            commitseqs = [torch.tensor(commitseq).float() for commitseq in commitseqs]
            commitseqsLength = torch.tensor([len(commitseq) for commitseq in commitseqs])
            commitseqsPadded = pad_sequence(commitseqs, batch_first=True)
            commitseqs = pack_padded_sequence(commitseqsPadded, commitseqsLength, batch_first=True, enforce_sorted=False)

        if(config.checkProcessMetricsExists()):
            processmetricss = torch.tensor(processmetricss).float()

        # yについて
        ys = torch.tensor(ys).float()
        return asts, astseqs, codemetricss, commitgraphs, commitseqs, processmetricss, ys