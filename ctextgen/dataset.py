import random

import torch
from torchtext import data, datasets
from torchtext.vocab import GloVe


class SST_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'

        train, val, test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
            filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1,
            shuffle=True, repeat=True
        )
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]


class IMDB_Dataset:
    def __init__(self):
        self.SEED = 1234
        self.MAX_VOCAB_SIZE = 25_000
        torch.manual_seed(self.SEED)
        self.BATCH_SIZE = 64

        self.TEXT = data.Field(tokenize='spacy', include_lengths=True)
        self.LABEL = data.LabelField(dtype=torch.float)

        train_data, test_data = datasets.IMDB.split(self.TEXT, self.LABEL)
        train_data, valid_data = train_data.split(random_state=random.seed(self.SEED))

        self.TEXT.build_vocab(train_data,
                         max_size=self.MAX_VOCAB_SIZE,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)

        self.LABEL.build_vocab(train_data)

        self.n_vocab = len(self.TEXT.vocab.itos)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.BATCH_SIZE,
            sort_within_batch=True,
            device=device)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]
