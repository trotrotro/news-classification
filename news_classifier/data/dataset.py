import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, df, word2idx, max_length=256):
        self.df = df
        self.word2idx = word2idx
        self.max_length = max_length

    def encode(self, text):
        tokens = text.split()

        ids = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

        ids = ids[: self.max_length]

        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids += [self.word2idx["<PAD>"]] * pad_len

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = self.encode(row["text"])
        y = torch.tensor(row["label"], dtype=torch.long)

        return x, y
