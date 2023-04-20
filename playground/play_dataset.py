import os
import sys
import dgl
import torch
import torch.nn as nn
import dgl.function as fn
import dgl.nn as dglnn
import torch.nn.functional as F
import pandas as pd

from dataset import MyDataset
from utils import read_dataset

import matplotlib.pyplot as plt
import networkx as nx

from model_gcn import Model


def _main():
    train_loop()


def play_visualization():
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    G = dgl.to_networkx(g)
    # pos = nx.spring_layout(G)
    options = {
        'node_color': 'skyblue',
        'edge_color': 'black',
        'node_size': 20,
        'width': 1,
        'with_labels': False,
    }
    nx.draw(G, **options)
    plt.show()


def check_nan():
    # dataset = MyDataset(feat_dim=20)
    # g = dataset.g
    # print(torch.any(torch.isnan(g.edata['score'])))

    nan_row_id = []
    df = pd.read_csv(os.path.join('../data/user_artist_play_raw.csv'))
    for i, row in df.iterrows():
        score = row['score']
        if torch.isnan(torch.tensor(score)):
            nan_row_id.append(i)
            print(row)
    print(len(nan_row_id))

    print(df.shape)
    df.dropna(inplace=True)
    print(df.shape)
    df.to_csv('../data/user_artist_play.csv')

    # df = df.drop(nan_row_id)
    #
    # nan_row_id = []
    # for i, row in df.iterrows():
    #     score = row['score']
    #     if torch.isnan(torch.tensor(score)):
    #         nan_row_id.append(row)
    # print(len(nan_row_id))


def train_loop():
    device = 'cpu'
    node_feat_dim = 20
    dataset = MyDataset(feat_dim=node_feat_dim)

    # Sampler
    E = dataset.g.number_of_edges()
    assert E % 2 == 0
    E /= 2
    reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)]).long()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
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
        batch_size=1024,  # #edges in one batch
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    valid_loader = dgl.dataloading.DataLoader(
        dataset.g,
        dataset.indices_valid,
        sampler,
        batch_size=1024,  # #edges in one batch
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_loader = dgl.dataloading.DataLoader(
        dataset.g,
        dataset.indices_test,
        sampler,
        batch_size=1024,  # #edges in one batch
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Model
    model = Model(in_features=node_feat_dim, hidden_features=20, out_features=20, num_classes=1)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    metric = nn.L1Loss()

    num_epochs = 10
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
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

            mae = metric(edge_predictions, edge_labels)
            mae_sum += mae

        num_batch = len(valid_loader)
        avg_valid_loss = total_loss / num_batch
        avg_valid_mae = mae_sum / num_batch

        print('Epoch: {:03d} | Train Loss: {:.4f}, MAE: {:.4f} | Valid Loss: {:.4f}, MAE: {:.4f}'.format(
            epoch, avg_train_loss, avg_train_mae, avg_valid_loss, avg_valid_mae
        ))

    # Test
    model.eval()
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
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

        mae = metric(edge_predictions, edge_labels)
        mae_sum += mae

    num_batch = len(test_loader)
    avg_test_loss = total_loss / num_batch
    avg_test_mae = mae_sum / num_batch
    print('Test result: Loss: {:.4f}, MAE: {:.4f}'.format(avg_test_loss, avg_test_mae))


def play_graph():
    u = torch.tensor([0, 1])
    v = torch.tensor([1, 2])
    e = torch.tensor([5, 10])
    g = dgl.graph((u, v))
    g.edata['weight'] = e
    print(g.edges[0, 1].data['weight'])
    print(g.edges[1, 2].data['weight'])

    g.ndata['feat'] = torch.rand(size=(3, 10))
    print(g.nodes[0].data['feat'])


def play_bigraph():
    u = torch.tensor([0, 1])
    v = torch.tensor([1, 2])
    e = torch.tensor([5, 10])
    g = dgl.graph((u, v))
    g.edata['weight'] = e
    print(g.edges[0, 1].data['weight'])
    print(g.edges[1, 2].data['weight'])

    g = dgl.graph(((torch.concat((u, v), dim=0)), (torch.concat((v, u), dim=0))))
    g.edata['weight'] = torch.concat((e, e), dim=0)
    print(g.num_nodes(), g.number_of_edges())
    print(g.edges[2, 1].data['weight'])

    # g = dgl.to_bidirected(g)
    # print(g.number_of_edges())
    # print(g.edata)
    # print(g.edges[0, 1].data['weight'])
    # print(g.edges[1, 2].data['weight'])


def _procedures():
    pass





if __name__ == '__main__':
    _main()
