import numpy as np
import tensorflow as tf


class LSTMFinder(tf.keras.Model):
    def __init__(self, graph, max_path_length, path_count):
        self.graph = graph
        self.rnn_stack = {}
        self.selection_mlp = {}
        self.history_depth = max_path_length
        self.prior_embedding_size = 2 * 200
        self.posterior_embedding_size = 3 * 200
        self.history_stack = tf.contrib.rnn.LSTMBlockCell(200, forget_bias=0.0)
        self.prior_mlp = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.prior_embedding_size),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(400, activation=tf.nn.relu)
        ])
        self.posterior_mlp = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.posterior_embedding_size),
            tf.keras.layers.Dense(400, activation=tf.nn.relu),
            tf.keras.layers.Dense(400, activation=tf.nn.relu)
        ])
        self.path_count = path_count

    def next_step_probabilities(self, history_stack_state, candidates, relation=None):
        # 根据relation计算prior特征或posterior特征
        if relation is None:
            feature = self.prior_mlp(history_stack_state)
        else:
            feature = self.posterior_mlp(history_stack_state + relation)

        # 计算点积
        choices = tf.keras.layers.dot(
            [candidates, feature]
        )

        # 计算选择概率
        probabilities = tf.nn.softmax(choices)
        return probabilities

    # prior/posterior, 通过relation区分
    def paths_between(self, from_node, to_node, relation=None):
        step = 0
        paths = []
        history_stack_states = []
        self.history_stack.zero_state(self.lstm_units, dtype=tf.float32)

        # 最大搜索history_depth跳
        while step < self.history_depth:
            # 选择邻接矩阵
            candidates = self.graph.neighbors_of(current_node)

            # 根据概率采样最终选择
            probabilities = self.next_step_probabilities(history_stack_states[0], candidates, relation)
            best_choice = tf.distributions.Categorical(probs=probabilities).sample().eval()

            paths[0] += candidates[best_choice]
            if candidates[best_choice].to_id == to_node:
                break

            current_node = candidates[best_choice].to_id
            input_vector = np.concatenate(
                (self.graph.vec_of_ent(current_node) + self.graph.vec_of_rel(candidates[best_choice].rel_id))
            )
            step = step + 1

            # 通过LSTM计算输出与状态
            _, history_stack_states[0] = self.history_stack(input_vector, history_stack_states[0])

        if paths[0][step - 1].to_node != to_node:
            return None
        else:
            return paths[0]
