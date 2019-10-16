from __future__ import division
from __future__ import print_function

import os
import time
import cvxopt
import tensorflow as tf

from multiprocessing import Pool
from utils import *
from models import GCN
from tqdm import trange
from scipy import sparse
from cvxopt import matrix

# Set random seed
seed = 123
np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('samples', 50, 'Number of samples to train.')
flags.DEFINE_integer('train_size', 20, 'The size of training dataset.')
flags.DEFINE_integer('test_size', 1000, 'The size of test dataset.')
flags.DEFINE_integer('max_order', 2, 'The max number of weight neighbor matrix to use.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_boolean('higher', True, 'Whether to increase the size of the convolution kernel.')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load data
adj, features, labels = load_data(FLAGS.dataset)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

add_adj = 0
if FLAGS.higher:
    for num_hop in range(2, FLAGS.max_order + 1):
        complex_file = 'complex_dir/%s_%s.npz' % (FLAGS.dataset, num_hop)
        if os.path.exists(complex_file):
            add_adj = sparse.load_npz(complex_file)
        else:
            cvxopt.solvers.options['show_progress'] = False
            add_adj = sp.lil_matrix((adj.shape[0], adj.shape[0]))
            improved_features = normalize_features(features)
            adj_features = normalize_adj(adj + sp.eye(adj.shape[0])).dot(improved_features)
            alpha_node = simple_solution(FLAGS.dataset, adj, improved_features, num_hop)
            dis_matrix = sp.csgraph.shortest_path(adj, directed=False, unweighted=True)
            row, col = np.where(dis_matrix == num_hop)[0], np.where(dis_matrix == num_hop)[1]
            care_num = [0] + [i for i in trange(1, row.shape[0]) if row[i] != row[i - 1]] + [row.shape[0]]


            def complex_solution(i):
                diff_feature_vectors = sp.vstack([improved_features[col[j]] - adj_features[row[j]]
                                                for j in range(care_num[i - 1], care_num[i])])
                diff_feature_matrix = diff_feature_vectors.dot(diff_feature_vectors.T)
                P = 2 * matrix(diff_feature_matrix.toarray().astype(np.double))
                q = matrix(np.zeros(diff_feature_matrix.shape[0], dtype=np.double))
                G = matrix(-np.eye(diff_feature_matrix.shape[0], dtype=np.double))
                h = matrix(np.zeros(shape=(diff_feature_matrix.shape[0], 1), dtype=np.double))
                A = matrix(np.ones(shape=(1, diff_feature_matrix.shape[0]), dtype=np.double),
                           (1, diff_feature_matrix.shape[0]))
                b = matrix(diff_feature_matrix.shape[0] * alpha_node[row[care_num[i - 1]]])
                sv = cvxopt.solvers.qp(P, q, G, h, A, b)
                return i, sv['x']

            pool = Pool(12)
            results = list(pool.map(complex_solution, range(1, len(care_num))))
            for k in trange(len(results)):
                i, sv_x = results[k]
                for j in range(care_num[i - 1], care_num[i]):
                    add_adj[row[j], col[j]] = sv_x[j - care_num[i - 1]]
            pool.close()
            pool.join()
            sparse.save_npz("complex_dir/%s_%s.npz" % (FLAGS.dataset, num_hop), add_adj.tocsr())

