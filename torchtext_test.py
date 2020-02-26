import torch
from torchtext import data, datasets
from torchtext.vocab import GloVe


class IMDB_Dataset:

    def __init__(self, emb_dim=50, mbsize=32):
        # self.TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
        # self.LABEL = data.LabelField(dtype = torch.float)

        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15 and ex.label != 'neutral'

        train, test = datasets.IMDB.splits(
            self.TEXT, self.LABEL
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.test_iter = data.BucketIterator.splits(
            (train, test), batch_size=mbsize, device=-1,
            shuffle=True, repeat=True
        )
        self.train_iter = iter(self.train_iter)
        self.test_iter = iter(self.test_iter)

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


if __name__ == '__main__':
    dataset = IMDB_Dataset()
    print("DONE: dataset is ready")

    inputs, labels = dataset.next_batch()
    print("DONE: iterator is ready")
