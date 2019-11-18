from __future__ import absolute_import, division, print_function

import sys

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import checkpoints as chk
from graph.graph import Graph
from loss_tools import type_to_one_hot
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from test_tools import predict_sample
from train_tools import even_types, show_type_distribution

emb_size = 100
beam = 5
max_path_length = 5
test_count = 320

database = 'weibo'
task = 'event_type'
task_dir_name = task.replace('/', '_').replace(':', '_')
reasoner_class = sys.argv[1]

graph = Graph(database + '.db')
graph.prohibit_relation(task)
rel_embs = {
    10: graph.vec_of_rel_name('entertainment'),
    12: graph.vec_of_rel_name('political')
}

checkpoint_dir = 'checkpoints/{}/{}/unified/{}/'.format(
    database,
    task_dir_name,
    reasoner_class)

prior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=True)
if reasoner_class == 'cnn':
    path_reasoner = CNNReasoner(graph=graph, emb_size=emb_size, max_path_length=max_path_length)
else:
    path_reasoner = GraphSAGEReasoner(graph=graph, emb_size=emb_size, neighbors=15)
path_reasoner_name = type(path_reasoner).__name__
print('using {}, {}'.format(path_reasoner_name, type(prior).__name__))

prior_checkpoint = tf.train.Checkpoint(model=prior)

likelihood_checkpoint = tf.train.Checkpoint(model=path_reasoner)

test_samples = graph.test_samples()[500:500+test_count]
# random.shuffle(test_samples)
show_type_distribution(test_samples)

for checkpoint_index in range(1, 11, 1):

    chk.load_from_index(
        prior_checkpoint,
        checkpoint_dir, 'prior',
        checkpoint_index
    )

    chk.load_from_index(
        likelihood_checkpoint,
        checkpoint_dir, path_reasoner_name,
        checkpoint_index
    )

    labels = []
    predicts = []
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    for sample in test_samples:
        label = type_to_one_hot(sample['rid'])
        rel_emb = rel_embs[sample['rid']]
        labels.append(label)

        predict = predict_sample(
            sample=sample,
            finder=prior,
            beam=beam,
            reasoner=path_reasoner,
            check_dest=True
        )
        predicts.append(predict)

    accuracy.update_state(
        labels,
        predicts
    )
    print('chk {} accuracy: {:.4f}'.format(
        checkpoint_index,
        accuracy.result().numpy()
    ))
