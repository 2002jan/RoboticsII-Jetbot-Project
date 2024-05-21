import copy

import torch
import torch.nn as nn
import torch.optim as optim
from math import ceil
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(
        model: nn.Module,
        epochs,
        train_dataloader,
        test_dataloader,
        device,
        patience=5,
        weights_init=False,
        lr=0.001
):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience)

    if weights_init:
        model.apply(init_weights)

    best_loss = float('inf')
    best_model_weights = None
    early_stopping_patience = ceil(patience * 1.5)

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        model.train(True)
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)

        running_vloss = 0.0

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(test_dataloader):
                v_inputs, v_labels = vdata[0].to(device), vdata[1].to(device)
                v_outputs = model(v_inputs)
                v_loss = loss_fn(v_outputs, v_labels)
                running_vloss += v_loss

        avg_v_loss = running_vloss / (i + 1)

        scheduler.step(avg_v_loss)
        print('LOSS train {} valid {} learning rate {}'.format(avg_loss, avg_v_loss,
                                                               scheduler.optimizer.param_groups[0]['lr']))

        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stopping_patience = ceil(patience * 1.5)
        elif avg_v_loss > (best_loss + 0.005):
            early_stopping_patience -= 1
            if early_stopping_patience <= 0:
                break

    model.load_state_dict(best_model_weights)


def train_one_epoch(model: nn.Module, train_dataloader, optimizer, loss_fn, device):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 49:
            last_loss = running_loss / 50
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


def get_model_size(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.01)
