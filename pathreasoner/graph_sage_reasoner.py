import tensorflow as tf

from layer.graphsage.aggregate import recursive
from layer.graphsage.layers import GraphConv, MaxPooling, NeighborSampler


class GraphSAGEReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size,
                 max_path_length=5,
                 aggregator='max_pooling', step_feature_width=None,
                 neighbors=None, random_sample=True):
        super(GraphSAGEReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.aggregators = []
        self.max_path_length = max_path_length
        self.mlp_width = 2 * emb_size

        if step_feature_width is None:
            step_feature_width = 2 * self.emb_size

        self.sampler = NeighborSampler(graph=graph, random_sample=random_sample)

        if aggregator == 'max_pooling':
            self.aggregator = MaxPooling(
                input_feature_dim=2 * self.emb_size,
                output_feature_dim=step_feature_width,
                neighbors=neighbors,
                dtype=tf.float32
            )
        elif aggregator == 'gcn':
            self.aggregator = GraphConv(
                input_feature_dim=2 * self.emb_size,
                output_feature_dim=step_feature_width,
                neighbors=neighbors,
                dtype=tf.float32)
        else:
            print('unknown aggregator param: ' + aggregator)

        self.path_stack = tf.keras.layers.LSTMCell(
            units=step_feature_width,
            dtype=tf.float32,
            kernel_regularizer=tf.keras.regularizers.l2(),
            bias_regularizer=tf.keras.regularizers.l2(),
            recurrent_regularizer=tf.keras.regularizers.l2()
        )

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, step_feature_width), dtype=tf.float32),
            tf.keras.layers.Dense(self.mlp_width, activation=tf.nn.relu,
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2()),
            tf.keras.layers.Dense(self.mlp_width, activation=tf.nn.relu,
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2()),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

    # likelihood
    def relation_of_path(self, path):
        step_feature = recursive(
            graph=self.graph,
            sampler=self.sampler,
            aggregators=[self.aggregator],
            root_id=path[0],
            skip=(path[2],),
            rel_id=None
        )
        stack_state = self.path_stack.get_initial_state(inputs=step_feature)

        for i in range(2, len(path), 2):
            if i < len(path) - 1:
                skip = (path[i - 2], path[i + 2])
            else:
                skip = (path[i - 2],)
            step_feature = recursive(
                graph=self.graph,
                sampler=self.sampler,
                aggregators=[self.aggregator],
                root_id=path[i],
                skip=skip,
                rel_id=path[i - 1]
            )
            output, stack_state = self.path_stack(
                inputs=step_feature,
                states=stack_state
            )

        probabilities = self.classifier(output)

        return probabilities[0]

