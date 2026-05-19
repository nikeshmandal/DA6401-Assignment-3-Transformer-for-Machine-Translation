import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def build_vocab(self, sentences, tokenizer):
        counter = Counter()
        for sent in sentences:
            tokens = [tok.text.lower() for tok in tokenizer(sent)]
            counter.update(tokens)
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

    def numericalize(self, sentence, tokenizer):
        tokens = [tok.text.lower() for tok in tokenizer(sentence)]
        return [self.sos_idx] + [self.stoi.get(tok, self.unk_idx) for tok in tokens] + [self.eos_idx]

    def __len__(self):
        return len(self.itos)


class Multi30kDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]['de']
        tgt = self.data[idx]['en']
        src_ids = self.src_vocab.numericalize(src, self.src_tokenizer)
        tgt_ids = self.tgt_vocab.numericalize(tgt, self.tgt_tokenizer)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    return src_batch, tgt_batch


def load_data_and_vocab(batch_size=128):
    from datasets import load_dataset
    from functools import partial

    print("Loading Multi30k dataset...")
    dataset = load_dataset("bentrevett/multi30k")

    print("Loading spacy models...")
    try:
        src_tokenizer = spacy.load("de_core_news_sm")
    except OSError:
        raise OSError("Run: python -m spacy download de_core_news_sm")

    try:
        tgt_tokenizer = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("Run: python -m spacy download en_core_web_sm")

    train_data = list(dataset['train'])
    val_data = list(dataset['validation'])
    test_data = list(dataset['test'])

    print("Building vocabularies...")
    src_vocab = Vocabulary(min_freq=2)
    tgt_vocab = Vocabulary(min_freq=2)

    src_vocab.build_vocab([x['de'] for x in train_data], src_tokenizer)
    tgt_vocab.build_vocab([x['en'] for x in train_data], tgt_tokenizer)

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    train_dataset = Multi30kDataset(train_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer)
    val_dataset = Multi30kDataset(val_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer)
    test_dataset = Multi30kDataset(test_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer)

    _collate = partial(collate_fn, src_pad_idx=src_vocab.pad_idx, tgt_pad_idx=tgt_vocab.pad_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer
