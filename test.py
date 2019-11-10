from __future__ import absolute_import, division, print_function

import sys

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import checkpoints as chk
from graph.graph import Graph
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from pathreasoner.learn import learn_from_path

emb_size = 100
beam = 5
max_path_length = 5
test_count = 50

database = sys.argv[1]
task = sys.argv[2]
task_dir_name = task.replace('/', '_').replace(':', '_')
reasoner_class = sys.argv[3]

graph = Graph(database + '.db')
graph.prohibit_relation(task)
checkpoint_dir = 'checkpoints/{}/{}/unified/{}/'.format(
    database,
    task_dir_name,
    reasoner_class)

posterior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=False)
prior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=True)
if reasoner_class == 'cnn':
    path_reasoner = CNNReasoner(graph=graph, emb_size=emb_size, max_path_length=max_path_length)
else:
    path_reasoner = GraphSAGEReasoner(graph=graph, emb_size=emb_size, neighbors=15)
path_reasoner_name = type(path_reasoner).__name__
print('using {}, {}, {}'.format(type(posterior).__name__, path_reasoner_name, type(prior).__name__))

posterior_checkpoint = tf.train.Checkpoint(model=posterior)
chk.load_latest_if_exists(
    posterior_checkpoint,
    checkpoint_dir, 'posterior'
)

prior_checkpoint = tf.train.Checkpoint(model=prior)
chk.load_latest_if_exists(
    prior_checkpoint,
    checkpoint_dir, 'prior'
)

likelihood_checkpoint = tf.train.Checkpoint(model=path_reasoner)
chk.load_latest_if_exists(
    likelihood_checkpoint,
    checkpoint_dir, path_reasoner_name
)

test_samples = graph.test_samples_of(task)
# random.shuffle(test_samples)
test_samples = test_samples[:test_count]

positive_rel_emb = graph.vec_of_rel_name(task)
negative_rel_emb = np.zeros(emb_size, dtype='f4')


def test(to_id, truth, paths):
    bad_path = 0
    losses = []
    for path in paths:
        if path[-1] != to_id:
            bad_path += 1

        diff, gradient = learn_from_path(path_reasoner, path, truth)
        losses.append(diff)
    return bad_path, losses


prior_losses = []
prior_fails = 0
posterior_losses = []
posterior_fails = 0


prior_losses = np.array(prior_losses)
posterior_losses = np.array(posterior_losses)
print('average prior loss: {:.4f}, posterior loss: {:.4f}'.format(
    np.average(prior_losses),
    np.average(posterior_losses)
))
print('prior bad: {}, posterior bad: {}'.format(prior_fails, posterior_fails))
