from math import floor
import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class TemporalNetworkDataset(Dataset):

    def __init__(self, path, generate_neg_examples=False, mode='train',
                 duplicate_examples=False, repeat_examples=False,
                 num_layers=2, self_loop=False, normalize_adj=False,
                 data_split=[0.50, 0.20, 0.05, 0.25]):
        """
        Parameters
        ----------
        path : str
            Path to the dataset file. For example, CollegeMsg.txt, etc.
        neg_examples: Boolean 
            Whether to generate negative examples or read from file. If True, then negative examples are generated and saved in the same directory as the dataset file as ${dataset file name}_neg_examples_${mode}.txt. If False, then examples are read from a file that should be named and located as described. Default: False.
        mode : str
            One of train, val or test. Default: train.
        duplicate_examples : Boolean
            Whether to keep multiple instances of an edge in the list of positive examples. Default: False.
        repeat_examples : Boolean
            Whether to keep a positive example that has already appeared been used for graph construction or training. Default: False.
        num_layers : int
            Number of layers in the computation graph. Default: 2.
        self_loop : Boolean
            Whether to add self loops. Default: False.
        normalize_adj : Boolean
            Whether to use symmetric normalization on the adjacency matrix. Default: False.
        data_split: list
            Fraction of edges to use for graph construction / train / val / test. Default: [0.85, 0.08, 0.02, 0.03].
        """
        super().__init__()

        self.path = path
        self.generate_neg_examples = generate_neg_examples
        self.mode = mode
        self.duplicate_examples = duplicate_examples
        self.repeat_examples = repeat_examples
        self.num_layers = num_layers
        self.self_loop = self_loop
        self.normalize_adj = normalize_adj
        self.data_split = data_split

        print('--------------------------------')
        print('Reading dataset from {}'.format(path))
        edges_all = self._read_from_file(path)
        print('Finished reading data.')

        print('Setting up graph.')
        vertex_id = {j : i for (i, j) in enumerate(np.unique(edges_all[:, :2]))}
        idxs = [floor(v*edges_all.shape[0]) for v in np.cumsum(data_split)]
        if mode == 'train':
            idx1, idx2 = idxs[0], idxs[1]
        elif mode == 'val':
            idx1, idx2 = idxs[1], idxs[2]
        elif mode == 'test':
            idx1, idx2 = idxs[2], idxs[3]
        edges_t, pos_examples = edges_all[:idx1, :], edges_all[idx1:idx2, :]

        edges_t[:, :2] = np.array([vertex_id[u] for u in edges_t[:, :2].flatten()]).reshape(edges_t[:, :2].shape)
        edges_s = np.unique(edges_t[:, :2], axis=0)

        self.n = len(vertex_id)
        self.m_s, self.m_t = edges_s.shape[0], edges_t.shape[0]

        adj = sp.coo_matrix((np.ones(self.m_s), (edges_s[:, 1], edges_s[:, 0])),
                            shape=(self.n,self.n),
                            dtype=np.float32)
        # adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if self_loop:
            adj += sp.eye(self.n)
        if normalize_adj:
            degrees = np.power(np.array(np.sum(adj, axis=1)), -0.5).flatten()
            degrees = sp.diags(degrees)
            adj = degrees.dot(adj.dot(degrees))

        self.adj = adj.tolil()
        self.nbrs_s = self.adj.rows
        self.features = torch.from_numpy(np.eye(self.n)).float()

        nbrs_t = [[] for _ in range(self.n)]
        for (u, v, t) in edges_t:
            nbrs_t[v].append((u, t))
        nbrs_t = np.array(nbrs_t)

        self.nbrs_t = nbrs_t
        print('Finished setting up graph.')

        print('Setting up examples.')
        if repeat_examples:
            pos_seen = set()
        else:
            pos_seen = set(tuple(e) for e in edges_s)
        pos_examples = pos_examples[:, :2]
        pos_examples = np.array([row for row in pos_examples \
                                 if (row[0] < self.n) and
                                 (row[1] < self.n) and
                                 ((row[0], row[1]) not in pos_seen)])
        if not duplicate_examples:
            pos_examples = np.unique(pos_examples, axis=0)

        neg_path = os.path.splitext(path)[0] + '_neg_examples_{}.txt'.format(mode)
        if not generate_neg_examples:
            neg_examples = np.loadtxt(neg_path)
        else:
            num_neg_examples = pos_examples.shape[0]
            neg_examples = []
            cur = 0
            n, _choice = self.n, np.random.choice
            neg_seen = set(tuple(e[:2]) for e in edges_all)
            while cur < num_neg_examples:
                u, v = _choice(n, 2, replace=False)
                if (u, v) in neg_seen:
                    continue
                cur += 1
                neg_examples.append([u, v])
            np.savetxt(neg_path, neg_examples)
        neg_examples = np.array(neg_examples, dtype=np.int64)

        x = np.vstack((pos_examples, neg_examples))
        y = np.concatenate((np.ones(pos_examples.shape[0]),
                            np.zeros(neg_examples.shape[0])))
        perm = np.random.permutation(x.shape[0])
        x, y = x[perm, :], y[perm]
        x, y = torch.from_numpy(x).long(), torch.from_numpy(y).long()
        self.x, self.y = x, y
        print('Finished setting up examples.')

        print('Dataset properties:')
        print('Mode: {}'.format(self.mode))
        print('Number of vertices: {}'.format(self.n))
        print('Number of static edges: {}'.format(self.m_s))
        print('Number of temporal edges: {}'.format(self.m_t))
        print('Number of examples/datapoints: {}'.format(self.x.shape[0]))
        print('--------------------------------')


    def _read_from_file(self, path):
        raise NotImplementedError

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def _form_computation_graph(self, idx):
        """
        Parameters
        ----------
        idx : int or list
            Indices of the node for which the forward pass needs to be computed.

        Returns
        -------
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        """
        _list, _set = list, set
        if type(idx) is int:
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            node_layers = [np.array(idx, dtype=np.int64)]

        for _ in range(self.num_layers):
            prev = node_layers[-1]
            arr = [node for node in prev]
            arr.extend([e[0] for node in arr for e in self.nbrs_t[node]])
            arr = np.array(_list(_set(arr)), dtype=np.int64)
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j : i for (i, j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset. An example is (edge, label).

        Returns
        -------
        edges : numpy array
            The edges in the batch.
        features : torch.FloatTensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            Each row is the list of neighbors of nodes in node_layers[0].
        labels : torch.LongTensor
            Labels (1 or 0) for the edges in the batch.
        """
        idx = list(set([v.item() for sample in batch for v in sample[0][:2]]))

        node_layers, mappings = self._form_computation_graph(idx)

        rows = self.nbrs_s[node_layers[0]]
        features = self.features[node_layers[0], :]
        labels = torch.FloatTensor([sample[1] for sample in batch])
        edges = np.array([sample[0].numpy() for sample in batch])
        edges = np.array([mappings[-1][v] for v in edges.flatten()]).reshape(edges.shape)
        
        # TODO: Pin memory. Change type of node_layers, mappings and rows to
        # tensor?

        return edges, features, node_layers, mappings, rows, labels

    def get_dims(self):
        return self.features.shape[0], 1

class CollegeMsg(TemporalNetworkDataset):

    def _read_from_file(self, path):
        return np.loadtxt(path, dtype=np.int64)

class BitcoinAlpha(TemporalNetworkDataset):

    def _read_from_file(self, path):
        edges_all = np.loadtxt(path, delimiter=',', dtype=np.int64)
        return np.concatenate((edges_all[:, :2], edges_all[:, 3:]), axis=1)

class FBForum(TemporalNetworkDataset):

    def _read_from_file(self, path):
        return np.loadtxt(path, delimiter=',', dtype=np.int64)

class IAContact(TemporalNetworkDataset):

    def _read_from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line.split('\t') for line in lines]
        lines = [[*line[0].split(), line[1].split()[1]] for line in lines]
        return np.array(lines, dtype=np.int64)

class IAContactsHypertext(TemporalNetworkDataset):

    def _read_from_file(self, path):
        return np.loadtxt(path, delimiter=',', dtype=np.int64)

class IAEnronEmployees(TemporalNetworkDataset):

    def _read_from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line.split() for line in lines]
        lines = [[line[0], line[1], line[3]] for line in lines]
        return np.array(lines, dtype=np.int64)

class IARadoslawEmail(TemporalNetworkDataset):

    def _read_from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line.split() for line in lines[2:]]
        lines = [[line[0], line[1], line[3]] for line in lines]
        return np.array(lines, dtype=np.int64)

class WikiElec(TemporalNetworkDataset):

    def _read_from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line.split() for line in lines[2:]]
        lines = [[line[0], line[1], line[3]] for line in lines]
        return np.array(lines, dtype=np.int64)