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
    task = 'concept:athletehomestadium'
    graph = Graph('graph.db')
    graph.prohibit_relation(task)
    samples = [
        ('-', [2592, 233, 16987, 275, 19365, 363, 3749]),
        ('+', [2592, 233, 16987, 119, 62111]),
        ('-', [48957, 254, 27732, 63, 2909, 36, 5099]),
        ('+', [56803, 226, 25762, 363, 20621]),
        ('-', [16862, 233, 54557, 82, 39045, 106, 34298]),
        ('+', [67621, 198, 62111, 387, 3749])
    ]
    tests = [
        ('+', [17194, 226, 46241, 119, 13360, 387, 53464]),
        ('-', [64681, 254, 47635, 278, 25313, 363, 48217])
    ]
    epoch = 5

    reasoners = [
        (GraphSAGEReasoner(graph=graph, emb_size=25, neighbors=25, step_feature_width=50),
         tf.optimizers.Adam(5e-3)),
        (CNNReasoner(graph=graph, emb_size=25, max_path_length=5),
         tf.optimizers.Adam(5e-3))
    ]

    for reasoner in reasoners:
        last_loss(reasoner[0], reasoner[1])
        test_loss(reasoner[0])
