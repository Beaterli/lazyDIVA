import numpy as np
import tensorflow as tf


# 极其辣鸡的前n个查找算法
def top_n_of_2d_choices(choices, n):
    positions = []
    for i in range(n):
        max_choice = 0.0
        max_x = -1
        max_y = -1
        for x, row in enumerate(choices):
            for y, choice in enumerate(row):
                if choice > max_choice and (x, y) not in positions:
                    max_choice = choice
                    max_x = x
                    max_y = y
        positions.append((max_x, max_y))
    return positions


# 测试前n个查找算法
test_choices = [
    [0.1, 0.6, 0.4],
    [0.3, 0.5, 0.9],
    [0.7, 0.8, 0.9]
]
print(top_n_of_2d_choices(test_choices, 5))


class LSTMFinder(tf.keras.Model):
    def __init__(self, graph, max_path_length):
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
    # 还没有测试过，很可能有bug
    def paths_between(self, from_id, to_id, width=5, relation=None):
        step = 0
        paths = [(from_id,)]
        history_stack_states = [self.history_stack.zero_state(self.lstm_units, dtype=tf.float32)]

        # 最大搜索history_depth跳
        while step < self.history_depth:
            all_candidates = [] * len(paths)
            all_probabilities = [] * len(paths)
            updated_paths = []
            updated_stack_states = []

            for index, path in enumerate(paths):
                # 跳过到达目的地的路径
                if path[-1] == to_id:
                    updated_paths.append(path)
                    updated_stack_states.append(history_stack_states[index])
                    continue

                # 选择邻接矩阵
                candidates = self.graph.neighbors_of(path[-1])
                all_candidates[index] = candidates

                # 计算邻接节点的概率
                probabilities = self.next_step_probabilities(history_stack_states[index], candidates, relation)
                all_probabilities[index] = probabilities

            # 将已到达终点的路径以外的候选路径筛选出来
            top_choices_index = top_n_of_2d_choices(all_probabilities, width - len(updated_paths))
            for _, index in enumerate(top_choices_index):
                path = paths[index[0]]
                next_step = all_candidates[index[0]][index[1]]
                # 添加新的top n的路径信息
                updated_paths.append(path + (next_step.rel_id, next_step.to_id))
                updated_stack_states.append(history_stack_states[index[0]])

                # 计算新的top n的LSTM 状态
                input_vector = np.concatenate(
                    (self.graph.vec_of_ent(next_step.rel_id) + self.graph.vec_of_rel(next_step.to_id))
                )
                _, updated_stack_states[-1] = self.history_stack(input_vector, updated_stack_states[-1])

            paths = updated_paths
            history_stack_states = updated_stack_states
            step = step + 1

        return paths
