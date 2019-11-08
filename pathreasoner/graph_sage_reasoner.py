import tensorflow as tf

from layer.graphsage.aggregate import directional
from layer.graphsage.layers import GraphConv


class GraphSAGEReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size):
        super(GraphSAGEReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.gcn_aggregator = GraphConv(
            root_feature_dim=self.emb_size,
            out_feature_dim=2 * self.emb_size,
            neighbor_feature_dim=2 * self.emb_size,
            dtype='f4'
        )
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, 2 * self.emb_size), dtype='f4'),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

    # likelihood
    def relation_of_path(self, path):
        path_feature = directional(
            graph=self.graph,
            aggregator=self.gcn_aggregator,
            path=path,
            width=2,
            max_neighbor=100)

        probabilities = self.classifier(path_feature)

        return probabilities[0]
