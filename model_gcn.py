import torch
import torch as th
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair


def _main():
    pass


def _procedures():
    pass


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, edge_as_weight, attn, augment):
        super().__init__()
        self.conv1 = GraphConv(in_features, hidden_features, allow_zero_in_degree=True, edge_as_weight=edge_as_weight,
                               attn=attn, augment=augment)
        self.conv2 = GraphConv(hidden_features, out_features, allow_zero_in_degree=True, edge_as_weight=edge_as_weight,
                               attn=attn, augment=augment)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x, edge_weight=blocks[0].edata['score']))
        x = F.relu(self.conv2(blocks[1], x, edge_weight=blocks[1].edata['score']))
        return x


class ScorePredictor(nn.Module):
    def __init__(self, num_classes, in_features):
        super().__init__()
        self.W = nn.Linear(2 * in_features, num_classes)

    def apply_edges(self, edges):
        data = torch.cat(([edges.src['x'], edges.dst['x']]), dim=1)
        return {'out': self.W(data)}

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['out']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_classes, edge_as_weight, attn, augment):
        super().__init__()
        self.gcn = StochasticTwoLayerGCN(in_features, hidden_features, out_features, edge_as_weight=edge_as_weight,
                                         attn=attn, augment=augment)
        self.predictor = ScorePredictor(num_classes, out_features)

    def forward(self, edge_subgraph, blocks, x):
        x = self.gcn(blocks, x)
        return self.predictor(edge_subgraph, x)


class GraphConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            norm='both',
            weight=True,
            bias=True,
            activation=None,
            allow_zero_in_degree=False,
            edge_as_weight=False,
            attn=False,
            augment=False,
    ):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise Exception('Invalid norm value. Must be either "none", "both", "right" or "left".'
                            ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.augment = augment
        self.edge_as_weight = edge_as_weight

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        self.attn = attn
        if attn:
            self.W = nn.Linear(in_feats, out_feats, bias=False)
            self.A = nn.Linear(2 * out_feats, 1, bias=False)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None):
        """
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise Exception('There are 0-in-degree nodes in the graph, ')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight

            weight = self.weight

            # For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            # Aggregate first then mult weight
            graph.srcdata['h'] = feat_src
            if self.edge_as_weight == False:
                graph.update_all(message_func=self.message_weightless, reduce_func=fn.sum(msg='m', out='h'))
            else:
                if self.attn == False and self.augment == False:
                    graph.update_all(message_func=self.u_mul_e, reduce_func=fn.sum(msg='m', out='h'))
                elif self.attn == True and self.augment == False:
                    graph.srcdata['W'] = self.W(graph.srcdata['h'])
                    graph.dstdata['W'] = self.W(feat_dst)
                    graph.update_all(message_func=self.message_attn, reduce_func=fn.sum(msg='m', out='h'))
                elif self.attn == True and self.augment == True:
                    graph.srcdata['W'] = self.W(graph.srcdata['h'])
                    graph.dstdata['W'] = self.W(feat_dst)
                    graph.update_all(message_func=self.message_attn_aug, reduce_func=fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            if weight is not None:
                rst = th.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def message_weightless(self, edges):
        message = edges.src['h']
        return {'m': message}

    def u_mul_e(self, edges):
        message = edges.src['h'] * edges.data['_edge_weight'].unsqueeze(1)
        return {'m': message}

    def message_attn(self, edges):
        W_i = edges.src['W']
        W_j = edges.dst['W']
        t = torch.cat([W_i, W_j], dim=1)
        attn = self.A(t)
        attn = F.softmax(attn, dim=0)

        message = edges.src['h'] * (edges.data['_edge_weight'].unsqueeze(1) + attn)
        return {'m': message}

    def message_attn_aug(self, edges):
        W_i = edges.src['W']
        W_j = edges.dst['W']
        t = torch.cat([W_i, W_j], dim=1)
        attn = self.A(t)
        attn = F.softmax(attn, dim=0)

        message = edges.src['h'] * (edges.data['_edge_weight'].unsqueeze(1) + attn +
                                    torch.normal(mean=0., std=0.1, size=edges.src['h'].shape, device='cpu'))
        return {'m': message}

    def augment_message(self, msg):
        msg['m'] = torch.rand(size=msg['m'].shape, device=msg['m'].device)
        return msg

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


if __name__ == '__main__':
    _main()
