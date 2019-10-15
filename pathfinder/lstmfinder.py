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
    def __init__(self, graph, emb_size, max_path_length):
        super(LSTMFinder, self).__init__()
        self.graph = graph
        self.rnn_stack = {}
        self.selection_mlp = {}
        self.history_depth = max_path_length
        self.emb_size = emb_size
        self.history_stack = tf.keras.layers.LSTM(emb_size)
        self.prior_mlp = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, 2 * emb_size)),
            tf.keras.layers.Dense(2 * emb_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(2 * emb_size, activation=tf.nn.relu)
        ])
        self.posterior_mlp = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, 3 * emb_size)),
            tf.keras.layers.Dense(3 * emb_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(2 * emb_size, activation=tf.nn.relu)
        ])

    def next_step_probabilities(self, ent_enb, history_stack_state, candidates, relation=None):
        # 根据relation计算prior特征或posterior特征

        if relation is None:
            feature = self.prior_mlp(
                np.concatenate((ent_enb, history_stack_state)))
        else:
            feature = self.posterior_mlp(
                np.expand_dims(
                    np.concatenate((ent_enb, history_stack_state, relation)),
                    axis=0
                )
            )

        emb_list = []
        for candidate in candidates:
            emb_list.append(np.concatenate((
                self.graph.vec_of_rel(candidate.rel_id),
                self.graph.vec_of_ent(candidate.to_id)
            )))
        candidates_emb = np.array(emb_list)

        # 计算点积
        choices = tf.keras.layers.dot(
            [candidates_emb, feature],
            axes=1
        )

        # 计算选择概率
        probabilities = tf.keras.activations.softmax(tf.reshape(choices, [1, len(candidates)]))
        return probabilities[0]

    # prior/posterior, 通过relation区分
    # 还没有测试过，很可能有bug
    def paths_between(self, from_id, to_id, relation=None, width=5):
        step = 0
        paths = [(from_id,)]
        self.history_stack.get_initial_state(inputs=np.zeros(self.emb_size))
        history_stack_states = [np.zeros(self.emb_size)]

        # 最大搜索history_depth跳
        while step < self.history_depth:
            all_candidates = [[]] * len(paths)
            all_probabilities = [[]] * len(paths)
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
                probabilities = self.next_step_probabilities(
                    self.graph.vec_of_ent(path[-1]),
                    history_stack_states[index],
                    candidates,
                    relation
                )
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
                    (self.graph.vec_of_rel(next_step.rel_id), self.graph.vec_of_ent(next_step.to_id))
                )
                _, updated_stack_states[-1] = self.history_stack(inputs=input_vector, states=updated_stack_states[-1])

            paths = updated_paths
            history_stack_states = updated_stack_states
            step = step + 1

        return paths
