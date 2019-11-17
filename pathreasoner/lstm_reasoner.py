import tensorflow as tf


class LSTMReasoner(tf.keras.Model):
    def __init__(self, graph, emb_size,
                 max_path_length=5,
                 step_feature_width=None):
        super(LSTMReasoner, self).__init__()
        self.graph = graph
        self.emb_size = emb_size
        self.max_path_length = max_path_length
        self.mlp_width = 2 * emb_size

        if step_feature_width is None:
            step_feature_width = 2 * self.emb_size

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
        step_feature = tf.expand_dims(
            tf.concat([tf.zeros(self.emb_size, dtype=tf.float32), self.graph.vec_of_ent(path[0])], axis=0),
            axis=0
        )
        stack_state = self.path_stack.get_initial_state(inputs=step_feature)

        for i in range(2, len(path), 2):
            step_feature = tf.expand_dims(
                tf.concat([self.graph.vec_of_ent(path[i - 1]), self.graph.vec_of_ent(path[i])], axis=0),
                axis=0
            )
            output, stack_state = self.path_stack(
                inputs=step_feature,
                states=stack_state
            )

        probabilities = self.classifier(output)

        return probabilities[0]
