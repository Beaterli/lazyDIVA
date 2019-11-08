import numpy as np
import tensorflow as tf

from graph.graph import Graph
from layer.graphsage.aggregate import directional
from layer.graphsage.layers import GraphConv
from loss import type_to_label
from pathreasoner.learn import learn_from_path


class GraphSAGEReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size, neighbors):
        super(GraphSAGEReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.aggregators = [
            GraphConv(
                input_feature_dim=2 * self.emb_size,
                output_feature_dim=1 * self.emb_size,
                neighbors=neighbors,
                dtype=tf.float32)]
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, 1 * self.emb_size), dtype=tf.float32),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

    # likelihood
    def relation_of_path(self, path):
        path_feature = directional(
            graph=self.graph,
            aggregators=self.aggregators,
            path=path)

        probabilities = self.classifier(tf.expand_dims(path_feature, axis=0))

        return probabilities[0]


if __name__ == '__main__':
    task = 'concept:athletehomestadium'
    graph = Graph('graph.db')
    graph.prohibit_relation(task)
    samples = [
        ('-', [2592, 233, 16987, 275, 19365, 363, 3749]),
        ('+', [2592, 233, 16987, 119, 62111])
    ]
    reasoner = GraphSAGEReasoner(graph=graph, emb_size=100, neighbors=15)
    optimizer = tf.optimizers.Adam(5e-4)
    for i in range(50):
        losses = []
        for (sample_type, path) in samples:
            label = type_to_label(sample_type)
            loss, gradient = learn_from_path(reasoner, path, label)
            optimizer.apply_gradients(zip(gradient, reasoner.trainable_variables))
            losses.append(loss)

        print('avg loss: ' + str(np.average(np.array(losses))))

    print('finished!')
