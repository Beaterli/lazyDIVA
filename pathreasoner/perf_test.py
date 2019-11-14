import numpy as np
import tensorflow as tf

import episodes as eps
from graph.graph import Graph
from loss_tools import type_to_one_hot
from pathreasoner.cnn_reasoner import CNNReasoner
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from pathreasoner.learn import learn_from_path


def last_loss(reasoner, optimizer):
    min_loss = 0.0
    for i in range(epoch):
        losses = []
        for (sample_type, path) in samples:
            label = type_to_one_hot(sample_type)
            loss, gradient = learn_from_path(reasoner, path, label)
            optimizer.apply_gradients(zip(gradient, reasoner.trainable_variables))
            losses.append(loss)

        min_loss = np.average(np.array(losses))
    print('{} train finished! min loss: {}'.format(type(reasoner).__name__, str(min_loss)))
    return min_loss


def test_loss(reasoner):
    losses = []
    for (sample_type, path) in tests:
        label = type_to_one_hot(sample_type)
        loss, gradient = learn_from_path(reasoner, path, label)
        losses.append(loss)
    avg = np.average(np.array(losses))
    print('{} test finished! avg loss: {}'.format(type(reasoner).__name__, str(avg)))
    return avg


if __name__ == '__main__':
    task = '/tv/tv_program/languages'
    graph = Graph('fb15k-237.db')
    graph.prohibit_relation(task)
    episodes = eps.load_previous_episodes('{}.json'.format(task.replace(':', '_').replace('/', '_')))
    samples = list(map(lambda e: (e['type'], e['paths'][0]), episodes[50:100]))
    tests = list(map(lambda e: (e['type'], e['paths'][0]), episodes[200:250]))
    epoch = 10
    emb_size = 100

    reasoners = [
        (GraphSAGEReasoner(graph=graph, emb_size=emb_size, neighbors=25, step_feature_width=2 * emb_size,
                           random_sample=True),
         tf.optimizers.Adam(1e-3)),
        (CNNReasoner(graph=graph, emb_size=emb_size, max_path_length=5),
         tf.optimizers.Adam(1e-3))
    ]

    for reasoner in reasoners:
        last_loss(reasoner[0], reasoner[1])
        test_loss(reasoner[0])
