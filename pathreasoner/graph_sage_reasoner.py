import tensorflow as tf

from layer.graphsage.aggregate import directional
from layer.graphsage.layers import GraphConv


class GraphSAGEReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size, neighbors):
        super(GraphSAGEReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.primary_aggregator = [
            GraphConv(
                input_feature_dim=2 * self.emb_size,
                output_feature_dim=2 * self.emb_size,
                neighbors=neighbors,
                dtype=tf.float32),
            GraphConv(
                input_feature_dim=2 * self.emb_size,
                output_feature_dim=1 * self.emb_size,
                neighbors=int(neighbors / 2),
                dtype=tf.float32
            )]
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

        probabilities = self.classifier(path_feature)

        return probabilities[0]
