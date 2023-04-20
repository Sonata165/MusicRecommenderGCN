import os
import sys
import dgl
import torch
import pandas as pd


def _main():
    pass


def _procedures():
    pass


def read_dataset():
    '''
    Read the user-artist-play_cnt data, return the graph.
    '''
    df = pd.read_csv('../data/user_artist_play.csv')
    u_id = torch.tensor(df['userID'])
    a_id = torch.tensor(df['artistID'])
    score = torch.tensor(df['score'])
    g = dgl.graph((u_id, a_id))
    g.edata['score'] = score
    return g


if __name__ == '__main__':
    _main()
