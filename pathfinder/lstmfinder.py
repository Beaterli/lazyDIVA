import tensorflow as tf


class LSTMFinder(tf.keras.Model):
    def __init__(self, graph, max_path_length, embedding_size, mlp_size):
        self.graph = graph
        self.rnn_stack = {}
        self.selection_mlp = {}
        self.history_depth = max_path_length
        self.embedding_size = embedding_size
        self.history_stack = tf.contrib.rnn.LSTMBlockCell(200, forget_bias=0.0)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=embedding_size),
            tf.keras.layers.Dense(mlp_size, activation=tf.nn.relu)
        ])

    # prior/posterior, 通过relation区分
    def path_between(self, from_node, to_node, relation=None):
        step = 0
        path = []
        current_link = from_node
        state = self.history_stack.zero_state(self.lstm_units, dtype=tf.float32)

        # 最大搜索history_depth跳
        while step < self.history_depth:
            # 通过LSTM计算输出与状态
            output, state = self.history_stack(current_link, state)

            # 根据relation计算prior特征或posterior特征
            if relation is None:
                feature = self.mlp(state)
            else:
                feature = self.mlp(state + relation)

            # 选择邻接矩阵
            candidates = self.graph.get_links_from(current_link)

            # 计算点积
            choices = tf.keras.layers.dot(
                [candidates, feature]
            )

            # 计算选择概率
            probabilities = tf.nn.softmax(choices)

            # 得出最终选择
            best_choice = tf.arg_max(probabilities, 0)

            path += candidates[best_choice]
            if candidates[best_choice].to_node == to_node:
                break

            current_link = candidates[best_choice].to_node
            step = step + 1

        if path[step - 1].to_node != to_node:
            return None
        else:
            return path
