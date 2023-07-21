class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels, alphabet="ACGT"):
        self.sequences = sequences
        self.labels = labels
        self.alphabet = alphabet
        self.n_classes = len(alphabet)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        one_hot_sequence = self.sequence_string_to_one_hot(sequence, self.alphabet)
        return one_hot_sequence, label

    def sequence_string_to_one_hot(self, seq, alphabet):
        conversion_dict = {c: i for (i, c) in enumerate(list(alphabet))}
        convert_fn = lambda c: conversion_dict[c]
        x = np.array([convert_fn(c) for c in seq])
        return np.eye(self.n_classes)[x].astype(np.float32).transpose(1, 0)
