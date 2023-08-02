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
    x_train=None,
    y_train=None,
    x_test=None,
    y_test=None,
    train_set=None,
    val_set=None,
    batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 0.01,
    num_classes: int = 2,
    pbar: bool = False,
) -> None:
    if train_set is None:
        train_dataset = tdata.TensorDataset(x_train, y_train)
    else:
        train_dataset = train_set
    if val_set is None:
        test_dataset = tdata.TensorDataset(x_test, y_test)
    else:
        test_dataset = val_set

    train_loader = tdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = tdata.DataLoader(test_dataset, batch_size=batch_size)

    optimizer = optim.Adam(maso_dn.parameters(), lr=lr)

    loss_fn = (
        nn.functional.cross_entropy
        if num_classes > 2
        else nn.functional.binary_cross_entropy
    )

    for epoch_idx in trange(n_epochs, leave=True, disable=True):
        maso_dn.train()
        it = tqdm(train_loader, leave=False, disable=not pbar)
        for batch in it:
            x, y = batch
            y_hat = maso_dn(x)
            # y_hat = torch.sigmoid(y_hat)
            loss = loss_fn(y_hat, y.squeeze() if num_classes > 2 else y)

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
                if num_classes > 2:
                    y_hat = torch.argmax(y_hat, dim=1)
                else:
                    y_hat = (y_hat > 0.5).to(torch.float)

                accuracy = torch.mean((y_hat == y).to(torch.float))
                accuracy = accuracy.item()
                total_accuracy.append(accuracy)
        total_accuracy = statistics.mean(total_accuracy)
        print(f"Epoch {epoch_idx + 1}/{n_epochs}; Accuracy {total_accuracy:.3f}")
