import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class GTMHA(nn.Module):
    """
    Multi-head Attention in Graph Transformer
    """

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)  # Number of nodes
        q = self.q_linear(h).reshape(self.num_heads, N, self.head_dim)
        q *= self.scaling
        k = self.k_linear(h).reshape(self.num_heads, N, self.head_dim)
        v = self.v_linear(h).reshape(self.num_heads, N, self.head_dim)

        k_t = k.transpose(2, 1)
        t = torch.bmm(q, k_t)  # Attention score
        attn = t * A  # Element-wise multiplication with edge
        attn = F.softmax(attn, dim=2)  # Normalize
        out = torch.bmm(attn, v)  # Weighted average with value

        return self.out_proj(out.reshape(N, -1))


class GTLayer(nn.Module):
    '''
    A Graph Transformer Layer
    '''

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.attn = GTMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.ff1 = nn.Linear(hidden_size, hidden_size * 2)
        self.ff2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.attn(A, h)
        h = self.bn1(h + h1)

        h2 = h
        h = self.ff2(F.relu(self.ff1(h)))
        h = h2 + h

        return self.bn2(h)


class SumNode(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        '''
        For a batched graph, sum all node embedding within each graph
        '''
        batched_graph = graph
        with batched_graph.local_scope():
            batched_graph.ndata['h'] = feat
            graphs = dgl.unbatch(batched_graph)
            ret = []
            for graph in graphs:
                node_sum = torch.sum(graph.ndata['h'], dim=0)
                ret.append(node_sum)
        ret = torch.stack(ret)
        return ret


class GTModelFeatDense(nn.Module):
    '''
    Graph Transformer model
    Receive node feature as input
    '''
    def __init__(
            self,
            out_size,
            hidden_size=80,
            pos_enc_size=2,
            num_layers=8,
            num_heads=8,
    ):
        super().__init__()
        self.proj = nn.Linear(4, hidden_size)
        self.pe_linear = nn.Linear(pos_enc_size, hidden_size)
        self.gt_layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.pooler = SumNode()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, g, X, pos_enc):
        edge_coordinates = torch.stack(g.edges())
        N = g.num_nodes()
        # A = dglsp.spmatrix(edge_coordinates, shape=(N, N))
        v = torch.ones(size=(edge_coordinates.shape[1],)).to(self.proj.weight.device)
        A = torch.sparse_coo_tensor(edge_coordinates, v, size=(N, N))
        A = A.to_dense()

        t1 = self.proj(X.float())
        t2 = self.pe_linear(pos_enc)
        h = t1 + t2

        for gt_layer in self.gt_layers:
            h = gt_layer(A, h)
        h = self.pooler(g, h)

        ret = self.mlp(h)
        return ret
