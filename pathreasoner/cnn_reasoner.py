import numpy as np
import tensorflow as tf


class CNNReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size, max_path_length):
        super(CNNReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.max_path_length = max_path_length
        self.cnn_windows = [
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=128, kernel_size=1,
                                       input_shape=(max_path_length, self.emb_size * 2),
                                       activation=tf.nn.relu,
                                       dtype='f4'),
                tf.keras.layers.MaxPool1D(max_path_length)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=128, kernel_size=2,
                                       input_shape=(max_path_length, self.emb_size * 2),
                                       activation=tf.nn.relu,
                                       dtype='f4'),
                tf.keras.layers.MaxPool1D(max_path_length - 1)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                                       input_shape=(max_path_length, self.emb_size * 2),
                                       activation=tf.nn.relu,
                                       dtype='f4'),
                tf.keras.layers.MaxPool1D(max_path_length - 2)
            ])
        ]
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, 384), dtype='f4'),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(2, activation=tf.nn.softmax),
        ])

    # likelihood
    def relation_of_path(self, path):
        # 输入
        input_mat = []
        for i in range(1, self.max_path_length * 2 + 1, 2):
            if i < len(path):
                input_mat.append(np.concatenate((self.graph.vec_of_rel(path[i]), self.graph.vec_of_ent(path[i + 1]))))
            else:
                input_mat.append(np.zeros(self.emb_size, dtype='f4'))

        cnn_input = np.expand_dims(np.array(input_mat), axis=0)

        # 计算特征
        feature_window_1 = self.cnn_windows[0](cnn_input)[0]
        feature_window_2 = self.cnn_windows[1](cnn_input)[0]
        feature_window_3 = self.cnn_windows[2](cnn_input)[0]
        concat_feature = tf.concat(
            [feature_window_1, feature_window_2, feature_window_3], 1
        )

        # 计算概率
        probabilities = self.classifier(concat_feature)
        return probabilities[0]

