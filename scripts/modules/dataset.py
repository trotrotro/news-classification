import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, df, word2idx, max_length=256):
        self.df = df
        self.word2idx = word2idx
        self.max_length = max_length

    def encode(self, text):
        tokens = text.split()

        ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

        # padding / truncation
        if len(ids) < self.max_length:
            ids += [self.word2idx["<PAD>"]] * (self.max_length - len(ids))
        else:
            ids = ids[: self.max_length]

        return torch.tensor(ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = self.encode(row["text"])
        y = torch.tensor(row["label"])

        return x, y
