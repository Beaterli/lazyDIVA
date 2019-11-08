from __future__ import absolute_import, division, print_function

import random

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import checkpoints as chk
import loss
from graph.graph import Graph
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner

epoch = 25
emb_size = 100
beam = 5
max_path_length = 5

task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)
checkpoint_dir = 'checkpoints/'

posterior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=False)
prior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=True)
path_reasoner = CNNReasoner(graph=graph, emb_size=emb_size * 2, max_path_length=max_path_length)

posterior_chkpt_file = 'checkpoints/unified/posterior_fine'
posterior_checkpoint = tf.train.Checkpoint(model=posterior)
chk.load_latest_if_exists(
    posterior_checkpoint,
    'checkpoints/unified/', 'posterior_fine'
)

prior_checkpoint = tf.train.Checkpoint(model=prior)
prior_chkpt_file = 'checkpoints/unified/prior'
chk.load_latest_if_exists(
    prior_checkpoint,
    'checkpoints/unified/', 'prior'
)

likelihood_checkpoint = tf.train.Checkpoint(model=path_reasoner)
likelihood_chkpt_file = 'checkpoints/unified/likelihood'
chk.load_latest_if_exists(
    likelihood_checkpoint,
    'checkpoints/unified/', 'likelihood'
)

test_samples = graph.test_samples_of(task)
random.shuffle(test_samples)
test_samples = test_samples[:50]

positive_rel_emb = graph.vec_of_rel_name(task)
negative_rel_emb = np.zeros(emb_size, dtype='f4')


def test(to_id, truth, paths):
    bad_path = 0
    losses = []
    for path in paths:
        if path[-1] != to_id:
            bad_path += 1

        diff, gradient = path_reasoner.learn_from_path(path, truth)
        losses.append(diff)
    return bad_path, np.array(losses)


for sample in test_samples:

    from_id = sample['from_id']
    to_id = sample['to_id']
    sample_type = sample['type']

    label = loss.type_to_label(sample['type'])
    if sample_type == '+':
        rel_emb = positive_rel_emb
    else:
        rel_emb = negative_rel_emb

    posterior_paths = list(map(
        lambda state: state.path,
        posterior.paths_between(
            from_id=from_id,
            to_id=to_id,
            relation=rel_emb,
            width=beam
        ))
    )
    prior_paths = list(map(
        lambda state: state.path,
        prior.paths_between(
            from_id=sample['from_id'],
            to_id=sample['to_id'],
            width=beam
        ))
    )

    prior_bads, prior_losses = test(to_id, label, prior_paths)
    print('prior: {} -> {}, {}: bad: {}, avg: {}'.format(
        from_id,
        to_id,
        sample_type,
        prior_bads,
        np.average(prior_losses)
    ))

    post_bads, post_losses = test(to_id, label, posterior_paths)
    print('posterior: {} -> {}, {}: bad: {}, avg: {}'.format(
        from_id,
        to_id,
        sample_type,
        post_bads,
        np.average(post_losses)
    ))
