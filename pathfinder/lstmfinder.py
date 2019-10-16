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
        if max_choice == 0.0:
            break
        positions.append((max_x, max_y))
    return positions


# 测试前n个查找算法
print(top_n_of_2d_choices(np.array([
    [0.1, 0.6, 0.4],
    [0.3, 0.5, 0.9],
    [0.7, 0.8, 0.9]
]), 5))
print(top_n_of_2d_choices(np.array([
    [0.1, 0.6, 0.4]
]), 5))


class LSTMFinder(tf.keras.Model):
    def __init__(self, graph, emb_size, max_path_length):
        super(LSTMFinder, self).__init__()
        self.graph = graph
        self.rnn_stack = {}
        self.selection_mlp = {}
        self.history_depth = max_path_length
        self.emb_size = emb_size
        self.history_stack = tf.keras.layers.LSTMCell(emb_size)
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

    def next_step_probabilities(self, ent_enb, history_vector, candidates, relation=None):
        # 根据relation计算prior特征或posterior特征

        if relation is None:
            feature = self.prior_mlp(
                np.concatenate((ent_enb, history_vector)))
        else:
            feature = self.posterior_mlp(
                np.expand_dims(
                    np.concatenate((ent_enb, history_vector, relation)),
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
        try:
            choices = tf.keras.layers.dot(
                [candidates_emb, feature],
                axes=1
            )
        except IndexError:
            print(candidates_emb.shape, feature.shape)

        # 计算选择概率
        probabilities = tf.keras.activations.softmax(tf.reshape(choices, [1, len(candidates)]))
        return probabilities[0]

    # prior/posterior, 通过relation区分
    # 还没有测试过，很可能有bug
    def paths_between(self, from_id, to_id, relation=None, width=5):
        step = 0
        paths = [(from_id,)]
        ent_of_paths = [{from_id}]

        initial_input = np.zeros(self.emb_size, dtype="f4")
        # [0]是hidden state, [1]是carry state
        initial_state = self.history_stack.get_initial_state(inputs=np.expand_dims(initial_input, axis=0))
        history_stack_states = [(initial_input, initial_state[0], initial_state[1])]

        # 最大搜索history_depth跳
        while step < self.history_depth:
            all_candidates = [[]] * len(paths)
            all_probabilities = [[]] * len(paths)
            updated_paths = []
            updated_stack_states = []
            updated_ent_of_paths = []

            for index, path in enumerate(paths):
                # 跳过到达目的地的路径
                if path[-1] == to_id:
                    updated_paths.append(path)
                    updated_stack_states.append(history_stack_states[index])
                    updated_ent_of_paths.append(ent_of_paths[index])
                    continue

                # 选择邻接矩阵
                candidates = []
                ent_in_path = ent_of_paths[index]
                for neighbor in self.graph.neighbors_of(path[-1]):
                    if neighbor.to_id not in ent_in_path:
                        candidates.append(neighbor)

                if len(candidates) == 0:
                    continue

                all_candidates[index] = candidates

                # 计算邻接节点的概率
                probabilities = self.next_step_probabilities(
                    self.graph.vec_of_ent(path[-1]),
                    history_stack_states[index][0],
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
                updated_ent_of_paths.append(ent_of_paths[index[0]].copy())
                updated_ent_of_paths[-1].add(next_step.to_id)
                updated_stack_states.append(history_stack_states[index[0]])

                # 计算新的top n的LSTM 状态
                input_vector = np.concatenate(
                    (self.graph.vec_of_rel(next_step.rel_id), self.graph.vec_of_ent(next_step.to_id))
                )

                output_vector, stack_state = self.history_stack(
                    inputs=np.expand_dims(input_vector, axis=0),
                    states=(updated_stack_states[-1][1], updated_stack_states[-1][2])
                )

                updated_stack_states[-1] = (output_vector[0], stack_state[0], stack_state[1])

            paths = updated_paths
            history_stack_states = updated_stack_states
            ent_of_paths = updated_ent_of_paths
            step = step + 1

        return paths
