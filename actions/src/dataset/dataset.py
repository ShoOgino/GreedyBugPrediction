import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        #commitデータについてはここでflatten

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]["x"], self.samples[index]["y"]

    def collate_fn_my(self, batch):
        print(batch)
        xs, ys = list(zip(*batch))
        xs = [torch.tensor(x).float() for x in xs]
        xsLength = torch.tensor([len(x) for x in xs])
        ys = torch.tensor(ys).float()
        xsPadded = pad_sequence(xs)
        xsPacked = pack_padded_sequence(xsPadded, xsLength, enforce_sorted=False)
        return xsPacked, ys
