import numpy as np
import tensorflow as tf

from graph.graph import Graph
from loss import type_to_label
from pathreasoner.cnn_reasoner import CNNReasoner
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from pathreasoner.learn import learn_from_path


def last_loss(reasoner, optimizer):
    min_loss = 0.0
    for i in range(epoch):
        losses = []
        for (sample_type, path) in samples:
            label = type_to_label(sample_type)
            loss, gradient = learn_from_path(reasoner, path, label)
            optimizer.apply_gradients(zip(gradient, reasoner.trainable_variables))
            losses.append(loss)

        min_loss = np.average(np.array(losses))
    print('{} train finished! min loss: {}'.format(type(reasoner).__name__, str(min_loss)))
    return min_loss


def test_loss(reasoner):
    losses = []
    for (sample_type, path) in tests:
        label = type_to_label(sample_type)
        loss, gradient = learn_from_path(reasoner, path, label)
        losses.append(loss)
    avg = np.average(np.array(losses))
    print('{} test finished! avg loss: {}'.format(type(reasoner).__name__, str(avg)))
    return avg


if __name__ == '__main__':
    task = '/film/director/film'
    graph = Graph('fb15k-237.db')
    graph.prohibit_relation(task)
    samples = [
        ('-', [6403, 441, 10211, 354, 4487]),
        ('+', [3818, 98, 5261]),
        ('-', [6850, 10, 6444]),
        ('+', [574, 415, 3853, 129, 12159]),
        ('-', [10564, 415, 4876, 403, 9838]),
        ('+', [7117, 249, 458, 370, 5612, 441, 355])
    ]
    tests = [
        ('+', [11211, 415, 3276, 403, 6452]),
        ('-', [11312, 98, 13872, 397, 5392, 87, 10927]),
        ('+', [5993, 98, 13403]),
        ('-', [2500, 21, 4527, 174, 8424])
    ]
    epoch = 10
    emb_size = 100

    reasoners = [
        (GraphSAGEReasoner(graph=graph, emb_size=emb_size, neighbors=25, step_feature_width=3 * emb_size),
         tf.optimizers.Adam(1e-3)),
        (CNNReasoner(graph=graph, emb_size=emb_size, max_path_length=5),
         tf.optimizers.Adam(1e-3))
    ]

    for reasoner in reasoners:
        last_loss(reasoner[0], reasoner[1])
        test_loss(reasoner[0])
