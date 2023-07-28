"""
Training script
"""
import statistics

import torch
import torch.optim as optim
import torch.nn as nn

import torch.utils.data as tdata

from tqdm import tqdm, trange

import maso


def train(
    maso_dn: maso.MASODN,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 0.01,
    pbar: bool = False,
) -> None:
    train_dataset = tdata.TensorDataset(x_train, y_train)
    test_dataset = tdata.TensorDataset(x_test, y_test)

    train_loader = tdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = tdata.DataLoader(test_dataset, batch_size=batch_size)

    optimizer = optim.SGD(maso_dn.parameters(), lr=lr)

    for epoch_idx in trange(n_epochs, leave=True, disable=True):
        maso_dn.train()
        it = tqdm(train_loader, leave=False, disable=not pbar)
        for batch in it:
            x, y = batch
            y_hat = maso_dn(x)
            y_hat = torch.sigmoid(y_hat)
            loss = nn.functional.binary_cross_entropy(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

            it.set_postfix({"loss": loss})
        total_accuracy = list()
        maso_dn.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                y_hat = maso_dn(x)
                y_hat = y_hat > 0

                accuracy = torch.mean((y_hat == y).to(torch.float))
                accuracy = accuracy.item()
                total_accuracy.append(accuracy)
        total_accuracy = statistics.mean(total_accuracy)
        print(f"Epoch {epoch_idx + 1}/{n_epochs}; Accuracy {total_accuracy:.3f}")
