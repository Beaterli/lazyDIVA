import tensorflow as tf

from layer.graphsage.aggregate import recursive
from layer.graphsage.layers import GraphConv, MaxPooling, NeighborSampler
from padding import zeros_bottom


class GraphSAGEReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size,
                 max_path_length=5,
                 cnn_feature_size=128,
                 aggregator='max_pooling', step_feature_width=None,
                 neighbors=None, random_sample=True):
        super(GraphSAGEReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.aggregators = []
        self.max_path_length = max_path_length

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

        self.cnn_windows = [
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=cnn_feature_size, kernel_size=2,
                                       input_shape=(max_path_length + 1, step_feature_width),
                                       activation=tf.nn.relu,
                                       dtype=tf.float32),
                tf.keras.layers.MaxPool1D(max_path_length)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=cnn_feature_size, kernel_size=3,
                                       input_shape=(max_path_length + 1, step_feature_width),
                                       activation=tf.nn.relu,
                                       dtype=tf.float32),
                tf.keras.layers.MaxPool1D(max_path_length - 1)
            ])
        ]

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, len(self.cnn_windows) * cnn_feature_size), dtype=tf.float32),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

    # likelihood
    def relation_of_path(self, path):
        features = []
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
            features.append(ent_feature[0])

        features = tf.stack(features)
        features = zeros_bottom(features, self.max_path_length + 1)

        cnn_input = tf.expand_dims(features, axis=0)

        # 计算特征
        windows = []
        for cnn in self.cnn_windows:
            windows.append(cnn(cnn_input)[0])
        concat_feature = tf.concat(windows, axis=1)

        probabilities = self.classifier(concat_feature)

        return probabilities[0]

