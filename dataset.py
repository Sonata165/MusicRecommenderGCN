import os
import sys
import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset


def _main():
    dataset = MyDataset()


def _procedures():
    pass


class MyDataset(DGLDataset):

    def __init__(
            self,
            feat_dim,
            url=None,
            raw_dir=None,
            save_dir=None,
            force_reload=False,
            verbose=False,
    ):
        self.feat_dim = feat_dim
        super().__init__(
            name='dataset_name',
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/user_artist_play.csv'))
        u_id = torch.tensor(df['userID'], dtype=torch.long)
        a_id = torch.tensor(df['artistID'], dtype=torch.long) + 2100
        score = torch.tensor(df['score'], dtype=torch.float)

        g = dgl.graph(((torch.concat((u_id, a_id), dim=0)), (torch.concat((a_id, u_id), dim=0))))
        g.edata['score'] = torch.concat((score, score), dim=0)
        self.g = g

        # Split the dataset
        num_edge = self.g.number_of_edges()
        num_test_edge = int(num_edge * 0.1)
        indices = torch.randperm(num_edge)
        self.indices_train = indices[:-(num_test_edge * 2)]
        self.indices_valid = indices[-(num_test_edge * 2):-num_test_edge]
        self.indices_test = indices[-num_test_edge:]

        train_mask = torch.zeros(num_edge, dtype=torch.bool)
        valid_mask = torch.zeros(num_edge, dtype=torch.bool)
        test_mask = torch.zeros(num_edge, dtype=torch.bool)
        train_mask[self.indices_train] = True
        valid_mask[self.indices_valid] = True
        test_mask[self.indices_test] = True

        # Create masks for splitting
        self.g.edata['train_mask'] = train_mask
        self.g.edata['valid_mask'] = valid_mask
        self.g.edata['test_mask'] = test_mask

        # Create initial node features
        # num_node = self.g.num_nodes()
        # init_feat = torch.rand(size=(num_node, self.feat_dim))
        # torch.nn.init.xavier_uniform_(init_feat)
        # self.g.ndata['features'] = init_feat

        # Some other way to initialize features
        self.g.ndata['features'] = self.g.in_degrees().view(-1, 1).float()
        # graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1)


    def __getitem__(self, idx):
        # get one example by index
        pass

    def __len__(self):
        # number of data examples
        pass

    if __name__ == '__main__':
        _main()
