import math
import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

class CollegeMsg(Dataset):

    def __init__(self, path, mode,
                 neg_examples_path='',
                 time=True,
                 num_layers=2,
                 self_loop=False, normalize_adj=False,
                 data_split=[0.5, 0.3, 0.05, 0.15]):
        """
        Parameters
        ----------
        path : str
            Path to the CollegeMsg.txt file.
        mode : str
            train / val / test
        neg_examples_path : str
            Path to file with negative examples, i.e., edges with label 0. If empty string, then negative examples are generated, default: ''.
        time : Boolean
            Whether to add timestamps to examples or not, default: True.
        num_layers : int
            Number of layers in the computation graph, default: 2.
        self_loop : Boolean
            Whether to add self loops, default: False.
        normalize_adj : Boolean
            Whether to use symmetric normalization on the adjacency matrix, default: False.
        splits : list
            Fraction of edges to use for graph construction / train / val / test, default: [0.5, 0.3, 0.05, 0.15].
        """
        super(CollegeMsg, self).__init__()

        # self.path = path
        # self.self_loop = self_loop
        # self.normalize_adj = normalize_adj
        self.mode = mode
        self.time = time
        self.num_layers = num_layers
        self.data_split = data_split 

        print('--------------------------------')
        print('Reading CollegeMsg dataset from {}'.format(path))
        edges_t = np.loadtxt(path, dtype=np.int64)
        print('Finished reading data.')

        print('Setting up data structures.')

        idxs = [math.floor(v * edges_t.shape[0]) for v in np.cumsum(data_split)]
        if mode == 'train':
            idx1, idx2 = idxs[0], idxs[1]
        elif mode == 'val':
            idx1, idx2 = idxs[1], idxs[2]
        elif mode == 'test':
            idx1, idx2 = idxs[2], idxs[3]
        edges_t, pos_examples = edges_t[:idx1, ], edges_t[idx1:idx2, :]

        vertex_id = {j : i for (i, j) in enumerate(np.unique(edges_t[:, :2]))}
        edges_t[:, :2] = np.array([vertex_id[u] for u in edges_t[:, :2].flatten()]).reshape(edges_t[:, :2].shape)
        edges_s = np.unique(edges_t[:, :2], axis=0)

        self.n = len(vertex_id)
        self.m_s, self.m_t = edges_s.shape[0], edges_t.shape[0]

        adj = sp.coo_matrix((np.ones(self.m_s), (edges_s[:, 0], edges_s[:, 1])),
                            shape=(self.n,self.n),
                            dtype=np.float32)
        # adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if self_loop:
            adj += sp.eye(self.n)
        if normalize_adj:
            degrees = np.power(np.array(np.sum(adj, axis=1)), -0.5).flatten()
            degrees = sp.diags(degrees)
            adj = (degrees.dot(adj.dot(degrees)))

        self.adj = adj.tolil()
        self.nbrs_s = self.adj.rows
        self.features = np.eye(self.n)

        timestamps = dict()
        for (u, v, t) in edges_t:
            if u not in timestamps.keys():
                timestamps[u] = dict()
            if v not in timestamps[u].keys():
                timestamps[u][v] = []
            timestamps[u][v].append(t)
            # if v not in timestamps.keys():
            #     timestamps[v] = dict()
            # if u not in timestamps[v].keys():
            #     timestamps[v][u] = []
            # timestamps[v][u].append(t)

        self.timestamps = timestamps

        nbrs_t = [[] for _ in range(self.n)]
        for (u, v, t) in edges_t:
            nbrs_t[u].append((v, t))
            # nbrs_t[v].append((u, t))
        nbrs_t = np.array(nbrs_t)

        self.nbrs_t = nbrs_t

        print('Finished setting up data structures.')

        print('Setting up examples.')
        if not time:
            pos_examples = pos_examples[:, :2]
        pos_examples = np.array([row for row in pos_examples if row[0] < self.n and row[1] < self.n])
        if neg_examples_path:
            neg_examples = np.loadtxt(neg_examples_path)
            neg_examples = np.array(neg_examples, dtype=np.int64)
        else:
            num_neg_examples = pos_examples.shape[0]
            neg_examples = []
            seen = set(tuple(e) for e in edges_s)
            cur = 0
            n, _choice = self.n, np.random.choice
            while cur < num_neg_examples:
                u, v = _choice(n, 2, replace=False)
                if (u, v) in seen:
                    continue
                cur += 1
                neg_examples.append([u, v])
            neg_examples = np.array(neg_examples, dtype=np.int64)
            if time:
                mn, mx = np.max(edges_t[:, 2])+1, np.max(pos_examples[:, 2])
                times = np.random.randint(mn, mx, size=neg_examples.shape[0])
                times = np.expand_dims(times, axis=1)
                times = np.array(times, dtype=np.int64)
                neg_examples = np.concatenate((neg_examples, times), axis=1)
            if time:
                save_path = os.path.join(os.path.dirname(path), 
                                     'CollegeMsg_neg_examples_time.txt')
            else:
                save_path = os.path.join(os.path.dirname(path), 
                                     'CollegeMsg_neg_examples.txt')
            np.savetxt(save_path, neg_examples)

        x = np.vstack((pos_examples, neg_examples))
        y = np.concatenate((np.ones(pos_examples.shape[0]),
                            np.zeros(neg_examples.shape[0])))
        perm = np.random.permutation(x.shape[0])
        x, y = x[perm, :], y[perm]
        x, y = torch.from_numpy(x).long(), torch.from_numpy(y).long()
        self.x, self.y = x, y

        print('Finished setting up examples.')
        print('--------------------------------')


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_dims(self):
        """
        Returns
        -------
        dimension of input features
        """
        return self.features.shape[1]

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset.

        Returns
        -------
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
        rows : list
        labels : torch.LongTensor
            An (n') length tensor of node labels.
        """
        idx = list(set([v.item() for sample in batch for v in sample[0][:2]]))
        times = []
        if self.time:
            times = list(set([sample[0][2].item() for sample in batch]))

        node_layers, mappings = self._form_computation_graph(idx, times)
        if self.time:
            rows = self.nbrs_s[node_layers[0]]
            features = self.features[node_layers[0], :]
            labels = self.y[node_layers[-1]]
            features = torch.FloatTensor(features)
            labels = torch.LongTensor(labels)
        else:
            rows = self.nbrs_s[node_layers[0]]
            features = self.features[node_layers[0], :]
            labels = self.y[node_layers[-1]]
            features = torch.FloatTensor(features)
            labels = torch.LongTensor(labels)

        return features, node_layers, mappings, rows, labels

    def _form_computation_graph(self, idx, times=[]):
        """
        Parameters
        ----------
        idx : int
            Index of the node for which the forward pass needs to be computed.

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

        if self.time:
            mn = np.min(times)
            for _ in range(self.num_layers):
                prev = node_layers[-1]
                arr = [node for node in prev]
                # TODO: figure out how to create arr.
                arr.extend([e[0] for node in arr for e in self.nbrs_t[node] if e[1] < mn])
                arr = np.array(_list(_set(arr)), dtype=np.int64)
                node_layers.append(arr)
        else:
            for _ in range(self.num_layers):
                prev = node_layers[-1]
                arr = [node for node in prev]
                arr.extend([v for node in arr for v in self.nbrs_s[node]])
                arr = np.array(_list(_set(arr)), dtype=np.int64)
                node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j : i for (i,j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings