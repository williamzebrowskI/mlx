import io
import itertools
import os
import zipfile
from urllib import request

import numpy as np


def load_local_dataset(data_dir):
    """
    Load a dataset from local text files.
    Assumes data_dir contains 'train.txt', 'valid.txt', 'test.txt'.
    """
    filenames = ["wiki.train.txt", "wiki.valid.txt", "wiki.test.txt"]
    return _load(data_dir, filenames)


def _load(save_dir, filenames):
    # *NB* First file is expected to be the training set
    with open(os.path.join(save_dir, filenames[0]), "r") as fid:
        vocab = set(t for l in fid.readlines() for t in l.strip().split(" "))
    eos = "<eos>"
    unk = "<unk>"  # Add unknown token
    vocab.add(eos)
    vocab.add(unk)
    vocab = {v: i for i, v in enumerate(vocab)}

    def to_array(dataset):
        with open(os.path.join(save_dir, dataset), "r") as fid:
            lines = (l.strip().split(" ") for l in fid.readlines())
        return np.array(
            [vocab.get(w, vocab[unk]) for line in lines for w in itertools.chain(line, [eos])],  # Use `get` to handle unknown words
            dtype=np.uint32,
        )

    datasets = [to_array(fn) for fn in filenames]
    return vocab, *datasets


# Updated main script to load the local dataset
# Specify the directory containing the local 'train.txt', 'valid.txt', 'test.txt'
data_directory = "/Users/williamzebrowski/Library/Mobile Documents/com~apple~CloudDocs/mlx/data"  # Replace with the path to your local data

# Load the dataset from local files
vocab, train, valid, test = load_local_dataset(data_directory)


# Now the rest of your training script can use the loaded train, valid, and test sets
# Here is the main training function from the previous script

import math
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Hyperparameters and configuration
use_gpu = True  # Set to True to use the Metal back-end
seed = 42  # Seed for the RNGs
context_size = 1024  # Context size in tokens of the model
num_blocks = 12  # Number of Transformer blocks
dim = 1024  # Dimensionality of embeddings and hidden layers
num_heads = 16  # Number of heads used for multi-head attention
checkpoint = True  # Perform gradient checkpointing
batch_size = 8  # Minibatch size
num_iters = 100000  # Iterations to train for
learning_rate = 3e-4  # AdamW learning rate
weight_decay = 1e-5  # Set the weight decay
lr_warmup = 200  # LR linear warmup iterations
steps_per_report = 10  # Number of training steps between loss reporting
steps_per_eval = 1000  # Number of training steps between validations
eval_test = False  # Evaluate on the test set after training


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def main():
    global batch_size, context_size, steps_per_eval, steps_per_report

    # Use vocab, train, valid, and test from local data
    # Initialize model:
    model = TransformerLM(
        len(vocab), num_blocks, dim, num_heads, checkpoint
    )
    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    optimizer = optim.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    def eval_fn(dataset):
        inputs, targets = map(mx.array, to_samples(context_size, dataset))
        loss = 0
        for s in range(0, targets.shape[0], batch_size):
            bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
            bx, by = map(mx.array, (bx, by))
            losses = loss_fn(model, bx, by, reduce=False)
            loss += mx.sum(losses).item()
        return loss / len(targets)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []
    tic = time.perf_counter()
    for it, (inputs, targets) in zip(range(num_iters), train_iterator):
        inputs, targets = map(mx.array, (inputs, targets))
        optimizer.learning_rate = min(1, it / lr_warmup) * learning_rate
        loss = step(inputs, targets)
        mx.eval(state)
        losses.append(loss.item())
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {steps_per_report / (toc - tic):.3f}"
            )
            losses = []
            tic = time.perf_counter()
        if (it + 1) % steps_per_eval == 0:
            val_loss = eval_fn(valid)
            toc = time.perf_counter()
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
                f"Val took {(toc - tic):.3f}s, "
            )
            tic = time.perf_counter()

    if eval_test:
        test_loss = eval_fn(test)
        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


if __name__ == "__main__":
    if not use_gpu:
        mx.set_default_device(mx.cpu)
    main()