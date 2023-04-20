import os
import sys
import dgl
import torch
import torch.nn as nn
from dgl.utils import expand_as_pair

from dataset import MyDataset

def _main():
    # network parameters
    net_parameters = {
        'input_dim': 20,
        'hidden_dim': 100,
        'output_dim': 8, # nb of classes
        'L': 2,
    }

    # instantiate network
    net = GatedGCN_Net(net_parameters)
    print(net)

    dataset = MyDataset(feat_dim=20)
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
    train_loader = dgl.dataloading.DataLoader(
        dataset.g,
        dataset.indices_train,
        sampler,
        batch_size=1024,  # #edges in one batch
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = net
    device = 'cpu'
    loss_func = nn.MSELoss()
    for input_nodes, edge_subgraph, blocks in train_loader:
        blocks = [b.to(device) for b in blocks]
        edge_subgraph = edge_subgraph.to(device)
        node_features = blocks[0].srcdata['features']
        edge_features = blocks[0].edata['features']
        edge_labels = edge_subgraph.edata['score']
        edge_predictions = model(edge_subgraph, blocks, node_features, edge_features)
        edge_predictions = edge_predictions.squeeze()
        loss = loss_func(edge_predictions, edge_labels)
        loss.backward()
        exit(10)

    # for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(train_loader):
    #     batch_x = batch_graphs.ndata['feat']
    #     batch_e = batch_graphs.edata['feat']
    #     batch_snorm_n = batch_snorm_n
    #     batch_snorm_e = batch_snorm_e
    #     batch_labels = batch_labels
    #     batch_scores = net.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)


def _procedures():
    pass


class MLP_layer(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb of hidden layers
        super(MLP_layer, self).__init__()
        list_FC_layers = [nn.Linear(input_dim, input_dim, bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = torch.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GatedGCN_layer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GatedGCN_layer, self).__init__()
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']  # e_ij = Ce_ij + Dhi + Ehj
        edges.data['e'] = e_ij
        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / torch.sum(sigma_ij, dim=1)  # hi = Ahi + sum_j eta_ij * Bhj
        return {'h': h}

    def forward(self, g, h, e):
        h_in = h  # residual connection
        e_in = e  # residual connection

        h_src, h_dst = expand_as_pair(h, g)
        g = dgl.block_to_graph(g)

        for node_type, node_feat in zip(['_N_src', '_N_dst'], [h_src, h_dst]):
            g.nodes[node_type].data['h'] = node_feat
            g.nodes[node_type].data['Ah'] = self.A(node_feat)
            g.nodes[node_type].data['Bh'] = self.B(node_feat)
            g.nodes[node_type].data['Dh'] = self.D(node_feat)
            g.nodes[node_type].data['Eh'] = self.E(node_feat)

            # g.ndata['h'] = h
            # g.ndata['Ah'] = self.A(h)
            # g.ndata['Bh'] = self.B(h)
            # g.ndata['Dh'] = self.D(h)
            # g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)

        g.update_all(self.message_func, self.reduce_func)

        for node_type in ['_N_src', '_N_dst']:
            h = g.ndata['h']  # result of graph convolution
            # h = h * snorm_n  # normalize activation w.r.t. graph node size
            h = self.bn_node_h(h)  # batch normalization
            h = torch.relu(h)  # non-linear activation
            h = h_in + h  # residual connection

        e = g.edata['e']  # result of graph convolution
        # e = e * snorm_e  # normalize activation w.r.t. graph edge size
        e = self.bn_node_e(e)  # batch normalization
        e = torch.relu(e)  # non-linear activation
        e = e_in + e  # residual connection

        return h, e


class GatedGCN_Net(nn.Module):

    def __init__(self, net_parameters):
        super(GatedGCN_Net, self).__init__()
        input_dim = net_parameters['input_dim']
        hidden_dim = net_parameters['hidden_dim']
        output_dim = net_parameters['output_dim']
        L = net_parameters['L']
        self.embedding_h = nn.Linear(input_dim, hidden_dim)
        self.embedding_e = nn.Linear(input_dim, hidden_dim)
        self.GatedGCN_layers = nn.ModuleList([GatedGCN_layer(hidden_dim, hidden_dim) for _ in range(L)])
        self.MLP_layer = MLP_layer(hidden_dim, output_dim)

    def forward(self, edge_graph, blocks, h, e):
        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        # graph convnet layers
        for GGCN_layer, block in zip(self.GatedGCN_layers, blocks):
            # h, e = GGCN_layer(g, h, e, snorm_n, snorm_e)
            h, e = GGCN_layer(block, h, e)

        # MLP classifier
        edge_graph.ndata['h'] = h
        y = dgl.mean_nodes(edge_graph, 'h')
        y = self.MLP_layer(y)

        return y

    def loss(self, y_scores, y_labels):
        loss = nn.CrossEntropyLoss()(y_scores, y_labels)
        return loss

    def accuracy(self, scores, targets):
        scores = scores.detach().argmax(dim=1)
        acc = (scores == targets).float().sum().item()
        return acc

    def update(self, lr):
        update = torch.optim.Adam(self.parameters(), lr=lr)
        return update




if __name__ == '__main__':
    _main()
