import tensorflow as tf

from layer.graphsage.aggregate import recursive
from layer.graphsage.layers import GraphConv, NeighborSampler


class GraphSAGEReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size,
                 vertical_mean=True, step_feature_width=None,
                 neighbors=None, random_sample=True):
        super(GraphSAGEReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.aggregators = []

        if step_feature_width is None:
            step_feature_width = 2 * self.emb_size

        self.sampler = NeighborSampler(graph=graph, random_sample=random_sample)

        self.aggregator = GraphConv(
            input_feature_dim=2 * self.emb_size,
            output_feature_dim=step_feature_width,
            neighbors=neighbors,
            vertical_mean=vertical_mean,
            dtype=tf.float32)

        self.step_lstm = tf.keras.layers.LSTMCell(
            units=step_feature_width,
            dtype=tf.float32
        )

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, step_feature_width), dtype=tf.float32),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

    # likelihood
    def relation_of_path(self, path):
        initial_input = tf.zeros(self.emb_size, tf.float32)
        # [0]是hidden state, [1]是carry state
        state = self.step_lstm.get_initial_state(inputs=tf.expand_dims(initial_input, axis=0))

        for i in range(0, len(path), 2):
            if i == 0:
                rel_id = None
            else:
                rel_id = path[i - 1]

            ent_feature = recursive(
                graph=self.graph,
                sampler=self.sampler,
                aggregators=[self.aggregator],
                root_id=path[i],
                rel_id=rel_id
            )
            output, state = self.step_lstm(
                inputs=ent_feature,
                states=state
            )

        path_feature = state[0]

        probabilities = self.classifier(path_feature)

        return probabilities[0]

