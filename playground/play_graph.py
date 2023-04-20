import os
import sys
import dgl
import torch
import pandas as pd

def _main():
    play_dataset()

def play_dataset():
    df = pd.read_csv('../data/user_artist_play.csv')
    u_id = torch.tensor(df['userID'])
    a_id = torch.tensor(df['artistID'])
    score = torch.tensor(df['score'])
    g = dgl.graph((u_id, a_id))
    g.edata['score'] = score


def play_heterograph():
    import dgl
    import torch as th

    # Create a heterograph with 3 node types and 3 edges types.
    graph_data = {
        ('user', 'is_friend', 'user'): (th.tensor([0, 1]), th.tensor([1, 2])),
        ('user', 'listen', 'artist'): (th.tensor([0, 0]), th.tensor([4, 1])),
        ('user', 'tag', 'artist'): (th.tensor([1]), th.tensor([0]))
    }
    # Note: all nodes in each type will be equal to [0, max_id], len=max_id+1
    g = dgl.heterograph(graph_data)
    print(g.ntypes)
    print(g.edges(etype='listen'))
    print(g.nodes(ntype='user'))
    print(g.etypes)
    print(g.canonical_etypes)

    g.nodes['artist'].data['name']=th.arange(5)
    print(g.nodes['artist'])

    g.nodes['listen']


def _procedures():
    pass


if __name__ == '__main__':
    _main()
