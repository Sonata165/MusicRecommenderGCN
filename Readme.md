# Code for CS6208 project
## Music Recommendation with Graph ConvNet
Ou Longshen's project code submission.

Task: edge regression

## Dataset
Last.fm -2k dataset: https://grouplens.org/datasets/hetrec-2011/
- 1,892 users
- 17,632 artists
- 92,834 pairs of user-artist playing event count
(Number of edges is too large for full batch training. Stochastic subgraph sampling is adopted.)

## Model
Graph Convolutional Networks (GCN) https://arxiv.org/abs/1609.02907

The implementation of graph convolutional layers in the code was inspired by and adapted from the Deep Graph Library (DGL): https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/graphconv.html#GraphConv

## Environment
    # On linux (CUDA 11.7)
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
    pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

## Files
    data            Dataset used in the experiment.
    data_analysis.py/ipynb  Data analysis and preparation
    models_gcn.py   GCN implementation and improvement
    dataset.py      Dataset class    
    train.py        Code for model training and evaluation.
    train.ipynb     Training log.
    utils.py        Util class and functions

## Run the code
Check the **train.ipynb** for previous training logs. 

Or use below command:
    
    python train.py
to run the training script. 