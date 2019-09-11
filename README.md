# GraphSAGE and Graph Attention Networks for Link Prediction
This is a PyTorch implementation of GraphSAGE from the paper [Inductive Representation Learning on Large Graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs)
and of Graph Attention Networks from the paper [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf). The code in this repository focuses on the link prediction task. Although the models themselves do not make use of temporal information, the datasets that we use are temporal networks obtained from [SNAP](http://snap.stanford.edu/data/index.html#temporal) and [Network Repository](https://networkrepository.com/dynamic.php).

This is just some code I have been playing around with - there may be issues or bugs. If you use this code and find them, let me know!

## Usage

In the `src` directory, edit the `config_gat.json` or `config_graphsage` file to specify arguments and
flags. Then run `python main.py --json config_{}.json`.

## Limitations

Although a nearly identical implementation of Graph Attention Networks performs well on the node classification task, I had trouble training them for the link prediction in these temporal networks. The  GraphSAGE model seems to work pretty well, so there might be a bug in my code, or inadequate hyperparameter search.

## References
* [Inductive Representation Learning on Large Graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs), Hamilton et al., NeurIPS 2017.
* [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf), Velickovic et al., ICLR 2018.
