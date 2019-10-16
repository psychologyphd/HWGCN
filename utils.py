import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import random
import sys
import os

from tqdm import trange


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Loads input data from gcn/data directory"""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def simple_solution(dataset, adj, improved_features, num_hop):
    dis_matrix = sp.csgraph.shortest_path(adj, directed=False, unweighted=True)
    add_adj = sp.lil_matrix((adj.shape[0], adj.shape[0]))
    file = "simple_dir/%s_alpha_%s.npy" % (dataset, num_hop)
    alpha_node = np.zeros(shape=adj.shape[0])
    if os.path.exists(file):
        alpha_node = np.load(file)
    else:
        adj_features = normalize_adj(adj + sp.eye(adj.shape[0])).dot(improved_features)
        loss_node = 1e9 * np.ones(shape=adj.shape[0])
        for i in trange(0, 100):
            add_adj[np.where(dis_matrix == num_hop)] = 0.01 * i
            diff_feature_matrix = add_adj.dot(improved_features) - adj_features
            loss_cur = diff_feature_matrix.dot(diff_feature_matrix.T).diagonal()
            for j in range(adj.shape[0]):
                if loss_cur[j] < loss_node[j]:
                    loss_node[j] = loss_cur[j]
                    alpha_node[j] = 0.01 * i
        np.save("simple_dir/%s_alpha_%s" % (dataset, num_hop), alpha_node)

    return alpha_node


def split_dataset(labels, train_size, test_size):
    
    label_pairs = np.where(labels == 1)
    label2nodes = {}

    for i in range(len(label_pairs[0])):
        node, label = label_pairs[0][i], label_pairs[1][i]
        if label not in label2nodes.keys():
            label2nodes[label] = [node]
        else:
            label2nodes[label].append(node)

    idx_train = []
    idx_test = []
    for key in label2nodes.keys():
        random.shuffle(label2nodes[key])
        idx_train += label2nodes[key][:train_size]
        idx_test += label2nodes[key][train_size:]

    random.shuffle(idx_test)
    idx_test = idx_test[:test_size]
    return idx_train, idx_test
