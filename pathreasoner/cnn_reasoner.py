import tensorflow as tf


class CNNReasoner(tf.keras.Model):
    def __init__(self, input_width, max_path_length):
        self.input_width = input_width
        self.max_path_length = max_path_length
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(max_path_length, input_width)),
        self.cnn_windows = [
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=128, kernel_size=[1, input_width], strides=(1, input_width),
                                       activation=tf.nn.relu),
                tf.keras.layers.MaxPool1D(max_path_length)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=128, kernel_size=[2, input_width], strides=(1, input_width),
                                       activation=tf.nn.relu),
                tf.keras.layers.MaxPool1D(max_path_length - 1)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=128, kernel_size=[3, input_width], strides=(1, input_width),
                                       activation=tf.nn.relu),
                tf.keras.layers.MaxPool1D(max_path_length - 2)
            ])
        ]
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(384, 1)),
            tf.keras.layers.Dense(400, activation=tf.nn.softmax),
        ])

    # likelihood
    def relation_of_path(self, path):
        # 输入
        cnn_input = self.input_layer(path)
        # 计算特征
        concat_feature = self.cnn_windows[0](cnn_input) + self.cnn_windows[1](cnn_input) + self.cnn_windows[2]
        # 计算概率
        probabilities = self.classifier(concat_feature)
        return probabilities
