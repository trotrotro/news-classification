from collections import Counter


def build_vocab(texts, min_freq: int = 2):
    """
    Builds vocabulary from training texts.

    Args:
        texts: Iterable with preprocessed texts.
        min_freq: Minimum word frequency to include in vocabulary.

    Returns:
        word2idx: Mapping from token to index.
        idx2word: Mapping from index to token.
    """
    counter = Counter()

    for text in texts:
        counter.update(text.split())

    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word
