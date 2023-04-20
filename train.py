'''
2023/4/18 Author: Longshen Ou
Train a GCN for recommender (edge regression)
'''
import os
import dgl
import torch
import torch.nn as nn
import numpy as np
import random
import warnings

from torch.optim.lr_scheduler import ExponentialLR

from dataset import MyDataset

from model_gcn import Model

# Seed
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
dgl.seed(seed)
warnings.filterwarnings("ignore", category=UserWarning)


def _main():
    # # Naive GCN
    # hparams = {
    #     'device': 'cuda:0',
    #     'num_layers': 2,
    #     'batch_size': 4096,
    #     'lr': 0.01,
    #     'decay': 1.0,
    #     'epochs': 20,
    #     'in_dim': 1,
    #     'd_model': 64,
    #     'edge_as_weight': False,
    #     'attn': False,
    #     'augment': False,
    # }

    # # Edge as weight
    # hparams = {
    #     'device': 'cuda:0',
    #     'num_layers': 2,
    #     'batch_size': 4096,
    #     'lr': 0.01,
    #     'decay': 1.0,
    #     'epochs': 20,
    #     'in_dim': 1,
    #     'd_model': 64,
    #     'edge_as_weight': True,
    #     'attn': False,
    #     'augment': False,
    # }
    #
    # # With attention
    # hparams = {
    #     # 'device': 'cpu',
    #     'device': 'cuda:0',
    #     'num_layers': 2,
    #     'batch_size': 4096,
    #     'lr': 0.01,
    #     'decay': 1.0,
    #     'epochs': 20,
    #     'in_dim': 1,
    #     'd_model': 64,
    #     'edge_as_weight': True,
    #     'attn': True,
    #     'augment': False,
    # }
    #
    # With augmentation
    hparams = {
        'device': 'cpu',
        # 'device': 'cuda:0',
        'num_layers': 2,
        'batch_size': 4096,
        'lr': 0.01,
        'decay': 0.8,
        'epochs': 20,
        'in_dim': 1,
        'd_model': 64,
        'edge_as_weight': True,
        'attn': True,
        'augment': True,
    }
    train(hparams)


def train(hparams):
    device = hparams['device']
    node_feat_dim = hparams['in_dim']
    dataset = MyDataset(feat_dim=node_feat_dim)

    # Sampler
    E = dataset.g.number_of_edges()
    assert E % 2 == 0
    E /= 2
    reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)]).long()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=hparams['num_layers'])
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude='reverse_id',
        reverse_eids=reverse_eids,
    )

    # Data loaders
    train_loader = dgl.dataloading.DataLoader(
        dataset.g,
        dataset.indices_train,
        sampler,
        batch_size=hparams['batch_size'],  # #edges in one batch
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )
    valid_loader = dgl.dataloading.DataLoader(
        dataset.g,
        dataset.indices_valid,
        sampler,
        batch_size=hparams['batch_size'],  # #edges in one batch
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )
    test_loader = dgl.dataloading.DataLoader(
        dataset.g,
        dataset.indices_test,
        sampler,
        batch_size=hparams['batch_size'],  # #edges in one batch
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # Model
    model = Model(
        in_features=node_feat_dim,
        hidden_features=hparams['d_model'],
        out_features=hparams['d_model'],
        num_classes=1,
        edge_as_weight=hparams['edge_as_weight'],
        attn=hparams['attn'],
        augment=hparams['augment'],
    )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hparams['lr'])
    scheduler = ExponentialLR(opt, gamma=hparams['decay'])
    loss_func = nn.MSELoss()  # MSE as loss, MAE as performance measure
    metric = nn.L1Loss()

    num_epochs = hparams['epochs']
    min_val_loss = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        mae_sum = 0.0
        for input_nodes, edge_subgraph, blocks in train_loader:
            blocks = [b.to(device) for b in blocks]
            edge_subgraph = edge_subgraph.to(device)
            input_features = blocks[0].srcdata['features']
            edge_labels = edge_subgraph.edata['score']
            edge_predictions = model(edge_subgraph, blocks, input_features)
            edge_predictions = edge_predictions.squeeze()
            loss = loss_func(edge_predictions, edge_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

            mae = metric(edge_predictions, edge_labels)
            mae_sum += mae

        num_batch = len(train_loader)
        avg_train_loss = total_loss / num_batch
        avg_train_mae = mae_sum / num_batch

        # Validation
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            mae_sum = 0.0
            for input_nodes, edge_subgraph, blocks in valid_loader:
                blocks = [b.to(device) for b in blocks]
                edge_subgraph = edge_subgraph.to(device)
                input_features = blocks[0].srcdata['features']
                edge_labels = edge_subgraph.edata['score']
                edge_predictions = model(edge_subgraph, blocks, input_features)
                edge_predictions = edge_predictions.squeeze()
                loss = loss_func(edge_predictions, edge_labels)
                total_loss += loss.item()

                mae = metric(edge_predictions, edge_labels)
                mae_sum += mae

        num_batch = len(valid_loader)
        avg_valid_loss = total_loss / num_batch
        avg_valid_mae = mae_sum / num_batch

        if avg_valid_loss < min_val_loss:
            min_val_loss = avg_valid_loss
            torch.save(model, 'checkpoint.pt')

        print('Epoch: {:03d} | Train Loss: {:.4f}, MAE: {:.4f} | Valid Loss: {:.4f}, MAE: {:.4f}'.format(
            epoch, avg_train_loss, avg_train_mae, avg_valid_loss, avg_valid_mae
        ))
        scheduler.step()

    # Test
    model = torch.load('checkpoint.pt')
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        mae_sum = 0.0
        for input_nodes, edge_subgraph, blocks in test_loader:
            blocks = [b.to(device) for b in blocks]
            edge_subgraph = edge_subgraph.to(device)
            input_features = blocks[0].srcdata['features']
            edge_labels = edge_subgraph.edata['score']
            edge_predictions = model(edge_subgraph, blocks, input_features)
            edge_predictions = edge_predictions.squeeze()
            loss = loss_func(edge_predictions, edge_labels)
            total_loss += loss.item()

            mae = metric(edge_predictions, edge_labels)
            mae_sum += mae

    num_batch = len(test_loader)
    avg_test_loss = total_loss / num_batch
    avg_test_mae = mae_sum / num_batch
    print('Test result: Loss: {:.4f}, MAE: {:.4f}'.format(avg_test_loss, avg_test_mae))


if __name__ == '__main__':
    _main()
