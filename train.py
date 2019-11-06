from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from multiprocessing import Pool
from osqp import OSQP
from utils import *
from models import GCN

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
flags.DEFINE_integer('val_size', 500, 'The size of validation dataset.')
flags.DEFINE_integer('test_size', 1000, 'The size of test dataset.')
flags.DEFINE_integer('max_order', 2, 'The max number of weight neighbor matrix to use.')
flags.DEFINE_integer('runs', 50, 'The size of random splits.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_boolean('higher', True, 'Whether to increase the size of the convolution kernel.')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load data
adj, features, labels = load_data(FLAGS.dataset)

add_adj = 0
weight_matrix = 0
path_finding_start = time.time()
dis_matrix = sp.csgraph.shortest_path(adj, directed=False, unweighted=True)
shortest_path_time = time.time() - path_finding_start
with open('time_log.txt', 'a') as f:
    f.write("Shortest path finding time for %s: %s\n" %(FLAGS.dataset, shortest_path_time))
if FLAGS.higher:
    for num_hop in range(2, FLAGS.max_order + 1):
        solution_file = 'solutions/%s_%s.npz' % (FLAGS.dataset, num_hop)
        if os.path.exists(solution_file):
            add_adj = sp.load_npz(solution_file)
        else:
            pretrain_start = time.time()
            num_nodes = adj.shape[0]
            add_adj = sp.lil_matrix((num_nodes, num_nodes))
            features = normalize_features(features)
            adj_features = normalize_adj(adj + sp.eye(num_nodes)).dot(features)
            row, col = np.where(dis_matrix == num_hop)[0], np.where(dis_matrix == num_hop)[1]
            care_num = [0] + [i for i in trange(1, row.shape[0]) if row[i] != row[i - 1]] + [row.shape[0]]
            nodes_alpha = np.zeros(num_nodes)
            for i in trange(1, len(care_num)):
                sum_feature = 0
                central_idx = row[care_num[i-1]]
                central_feature = adj_features[central_idx]
                for j in range(care_num[i-1], care_num[i]):
                    sum_feature += features[col[j]]
                nodes_alpha[central_idx] = sum_feature.dot(central_feature.T) / sum_feature.dot(sum_feature.T)

            
            def matrix_solutions(i):
                diff_feature_vectors = sp.vstack([features[col[j]] - adj_features[row[j]]
                                                for j in range(care_num[i - 1], care_num[i])])
                diff_feature_matrix = diff_feature_vectors.dot(diff_feature_vectors.T)
                P = 2 * diff_feature_matrix
                q = np.zeros(diff_feature_matrix.shape[0], dtype=np.double)
                G = -np.eye(diff_feature_matrix.shape[0], dtype=np.double)
                h = np.zeros(shape=(diff_feature_matrix.shape[0]), dtype=np.double)
                A = np.ones(shape=(diff_feature_matrix.shape[0]), dtype=np.double)
                b = diff_feature_matrix.shape[0] * nodes_alpha[row[care_num[i - 1]]]
                P = sp.csc_matrix(P)
                A = sp.csc_matrix(A)
                l = -np.inf * np.ones(len(h))
                qp_A = sp.vstack([G, A]).tocsc()
                qp_l = np.hstack([l, b])
                qp_u = np.hstack([h, b])
                osqp = OSQP()
                osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False)
                res = osqp.solve()
                return i, res.x

            pool = Pool(os.cpu_count())
            results = list(pool.map(matrix_solutions, range(1, len(care_num))))
            pool.close()
            pool.join()
            for k in trange(len(results)):
                i, sv_x = results[k]
                for j in range(care_num[i - 1], care_num[i]):
                    add_adj[row[j], col[j]] = sv_x[j - care_num[i - 1]]
            pretrain_time = time.time() - pretrain_start
            with open('time_log.txt', 'a') as f:
                f.write("Pretraining time for %s of %s-th order matrix: %s\n" %(FLAGS.dataset, num_hop, pretrain_time))
            sp.save_npz("solutions/%s_%s.npz" % (FLAGS.dataset, num_hop), add_adj.tocsr())
        weight_matrix += add_adj

all_train_mask, all_val_mask, all_test_mask = [], [], []
all_y_train, all_y_val, all_y_test = [], [], [] 
for i in range(FLAGS.runs):
    idx_train, idx_val, idx_test = split_dataset(labels, FLAGS.train_size, FLAGS.val_size, FLAGS.test_size)
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    all_train_mask.append(train_mask)
    all_val_mask.append(val_mask)
    all_test_mask.append(test_mask)
    all_y_train.append(y_train)
    all_y_val.append(y_val)
    all_y_test.append(y_test)

features = preprocess_features(features)
support = [preprocess_adj(weight_matrix + adj)]
num_supports = 1

train_start = time.time()
for i in trange(FLAGS.runs, desc='Random splits'):
    train_mask, val_mask, test_mask = all_train_mask[i], all_val_mask[i], all_test_mask[i]
    y_train, y_val, y_test = all_y_train[i], all_y_val[i], all_y_test[i]

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = GCN(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()


    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1]

    # Init variables
    sess.run(tf.global_variables_initializer())

    max_val_acc = 0
    test_acc = 0
    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        _, acc = evaluate(features, support, y_val, val_mask, placeholders)
        if acc > max_val_acc:
            max_val_acc = acc
            # Testing
            _, test_acc = evaluate(features, support, y_test, test_mask, placeholders)


    print("Test set results:", "accuracy=", "{:.5f}".format(test_acc))

run_time = time.time() - train_start

with open('time_log.txt', 'w') as f:
    f.write("Run time for %s of %s runs: %s\n" %(FLAGS.cora, FLAGS.runs, run_time))