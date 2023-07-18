import os
import subprocess

def download_data(savedir="./"):
    """
    Arguments
    ---------
    savedir <str> - path (relative to the current working directory) where
                    the downloaded dataset is to be saved.

    Returns
    ------
    None
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    download_links = [
        'https://www.dropbox.com/s/drnyowfdv1lbjz6/train_sequences.txt',
        'https://www.dropbox.com/s/1voerf29dhakmqi/valid_sequences.txt',
        'https://www.dropbox.com/s/sq8wpx6ogsi7l2w/test_sequences.txt',
        'https://www.dropbox.com/s/8td0n60zb6bdi12/train_labels.txt',
        'https://www.dropbox.com/s/8drv3039bg837kx/valid_labels.txt',
        'https://www.dropbox.com/s/f31qxggnfqjddb9/test_labels.txt',
                ]
    for url in download_links:
        cmd = ["wget", "-P", f"{savedir}", f"{url}"]
        _ = subprocess.call(cmd)

def load_data(savedir="./"):
    """
    Arguments
    ---------
    savedir <str> - path (relative to the current working directory) where
                    the downloaded dataset is saved.

    Returns
    ------
    Xs <dict>, Ys <dict> - Dictionary of input sequences and target labels. The
                        keys of the dictionary represent the training, validation
                        and test splits.
    """
    Xs, Ys = {}, {}
    splits = ['train', 'valid', 'test']
    for split in splits:
        with open(os.path.join(savedir, f"{split}_sequences.txt"), "r") as f:
            seqs = f.read().strip().split("\n")
            Xs[split] = seqs
        Ys[split] = np.loadtxt(os.path.join(savedir, f"{split}_labels.txt")).astype(np.float32)
    return Xs, Ys

def sequence_string_to_one_hot(seqs, alphabet="ACGT"):
    """
    Arguments
    ---------
    seqs <list of <str>> - A list of DNA sequences.
    alphabet <str of length 4> - The ordering of the base pairs, default - ACGT.


    Returns
    -------
    one_hot_seqs <numpy.ndarray> - A numpy array consisting of the one hot representation
    of the input sequences.
    """
    assert isinstance(seqs, list), "pass a list of sequences."
    assert len(alphabet) == 4
    conversion_dict = {c:i for (i, c) in enumerate(list(alphabet))}
    convert_fn = lambda c : conversion_dict[c]
    X = []
    for seq in seqs:
        x = np.array([convert_fn(c) for c in seq])
        X.append(x)
    return np.eye(4)[np.array(X)].astype(np.float32).transpose(0, 2, 1)