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
    return min_loss


if __name__ == '__main__':
    task = 'concept:athletehomestadium'
    graph = Graph('graph.db')
    graph.prohibit_relation(task)
    samples = [
        ('-', [2592, 233, 16987, 275, 19365, 363, 3749]),
        ('+', [2592, 233, 16987, 119, 62111])
    ]
    epoch = 20

    print('graph sage finished! min loss: {}'.format(str(last_loss(
        GraphSAGEReasoner(graph=graph, emb_size=100, neighbors=15, width=1),
        tf.optimizers.Adam(5e-3)))))
    print('cnn finished! min loss: {}'.format(str(last_loss(
        CNNReasoner(graph=graph, emb_size=100, max_path_length=5),
        tf.optimizers.Adam(5e-3)))))
