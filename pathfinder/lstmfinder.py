import numpy as np
import tensorflow as tf
from finderstate import FinderState


class LSTMFinder(tf.keras.Model):
    def __init__(self, graph, emb_size, max_path_length):
        super(LSTMFinder, self).__init__()
        self.graph = graph
        self.rnn_stack = {}
        self.selection_mlp = {}
        self.history_depth = max_path_length
        self.emb_size = emb_size
        self.history_stack = tf.keras.layers.LSTMCell(2 * emb_size)
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
    def paths_between(self, from_id, to_id, width=5, relation=None):
        step = 0

        initial_input = np.zeros(self.emb_size, dtype="f4")
        # [0]是hidden state, [1]是carry state
        initial_state = self.history_stack.get_initial_state(inputs=np.expand_dims(initial_input, axis=0))
        states = [FinderState(from_id, (initial_input, initial_state[0], initial_state[1]))]

        # 最大搜索history_depth跳
        while step < self.history_depth:
            all_candidates = []
            action_probs = []
            flatten_probs = []
            history_states = []
            chosen_to_pos = []
            updated_states = []
            tapes = []

            for index, state in enumerate(states):
                # 跳过到达目的地的路径
                path = state.path
                current_id = path[-1]
                if current_id == to_id:
                    updated_states.append(state)
                    continue

                # 选择邻接矩阵
                candidates = []
                for neighbor in self.graph.neighbors_of(current_id):
                    if neighbor.to_id not in state.entities:
                        candidates.append(neighbor)

                if len(candidates) == 0:
                    continue

                with tf.GradientTape() as tape:
                    # 计算新的top n的LSTM状态
                    rel_vector = np.zeros(self.emb_size)
                    if len(path) > 1:
                        rel_vector = self.graph.vec_of_rel(path[-2])
                    input_vector = np.concatenate(
                        (rel_vector, self.graph.vec_of_ent(current_id))
                    )

                    history_state = state.history_state
                    output_vector, stack_state = self.history_stack(
                        inputs=np.expand_dims(input_vector, axis=0),
                        states=(history_state[1], history_state[2])
                    )

                    history_state = (output_vector[0], stack_state[0], stack_state[1])
                    history_states.append(history_state)

                    all_candidates.append(candidates)

                    # 计算邻接节点的概率
                    probabilities = self.next_step_probabilities(
                        self.graph.vec_of_ent(current_id),
                        history_state[0],
                        candidates,
                        relation
                    )
                    action_probs.append(probabilities)
                    flatten_probs = flatten_probs + probabilities
                    for action_index in range(len(probabilities)):
                        chosen_to_pos.append((index, action_index))

                    tapes.append(tape)

            # 从动作空间中随机选取n个动作
            actions_chosen = np.random.choice(np.arange(len(flatten_probs)), size=width, replace=False, p=flatten_probs)
            # 根据选取的动作更新路径状态
            for action_chosen in actions_chosen:
                state_index = chosen_to_pos[action_chosen][0]
                candidate_index = chosen_to_pos[actions_chosen][1]
                updated_states.append(FinderState(
                    path_step=all_candidates[state_index][candidate_index],
                    history_state=history_states[state_index],
                    action_prob=action_probs[state_index],
                    action_chosen=candidate_index,
                    tape=tapes[state_index],
                    prev_state=states[state_index]
                ))

            states = updated_states
            step = step + 1

        return states
