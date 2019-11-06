import numpy as np
import tensorflow as tf

import loss
from pathfinder.decision import pick_top_n, index_of
from pathfinder.finderstate import FinderState


class LSTMFinder(tf.keras.Model):
    def __init__(self, graph, emb_size, max_path_length, prior=False):
        super(LSTMFinder, self).__init__()
        self.graph = graph
        self.rnn_stack = {}
        self.selection_mlp = {}
        self.history_depth = max_path_length
        self.emb_size = emb_size
        self.history_width = 2 * emb_size
        self.history_stack = tf.keras.layers.LSTMCell(
            units=self.history_width,
            kernel_regularizer=tf.keras.regularizers.l2(),
            bias_regularizer=tf.keras.regularizers.l2(),
            recurrent_regularizer=tf.keras.regularizers.l2()
        )
        if prior:
            self.mlp = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(1, self.history_width + 1 * emb_size)),
                tf.keras.layers.Dense(
                    units=2 * emb_size,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    bias_regularizer=tf.keras.regularizers.l2()
                ),
                tf.keras.layers.Dense(
                    units=2 * emb_size,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    bias_regularizer=tf.keras.regularizers.l2()
                )
            ])
        else:
            self.mlp = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(1, self.history_width + 2 * emb_size)),
                tf.keras.layers.Dense(
                    2 * emb_size,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    bias_regularizer=tf.keras.regularizers.l2()
                ),
                tf.keras.layers.Dense(
                    2 * emb_size,
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    bias_regularizer=tf.keras.regularizers.l2()
                )
            ])

    def initial_state(self, from_id):
        initial_input = np.concatenate((np.zeros(self.emb_size, dtype="f4"), self.graph.vec_of_ent(from_id)))
        # [0]是hidden state, [1]是carry state
        initial_state = self.history_stack.get_initial_state(inputs=np.expand_dims(initial_input, axis=0))
        return FinderState(path_step=from_id, history_state=(initial_input, initial_state[0], initial_state[1]))

    def available_action_probs(self, state, label_vec=None):
        path = state.path
        current_id = path[-1]
        # 选择邻接矩阵
        candidates = self.graph.neighbors_of(current_id)

        if len(candidates) == 0:
            return [], np.zeros(0, dtype='f4'), state.history_state

        # 计算LSTM状态
        rel_vector = np.zeros(self.emb_size, dtype='f4')
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

        ent_vec = self.graph.vec_of_ent(current_id)

        # 根据relation计算prior特征或posterior特征
        if label_vec is not None:
            mlp_input = tf.concat([ent_vec, history_state[0], label_vec], 0)
        else:
            mlp_input = tf.concat([ent_vec, history_state[0]], 0)
        feature = self.mlp(tf.expand_dims(mlp_input, axis=0))

        emb_list = []
        for candidate in candidates:
            emb_list.append(np.concatenate((
                self.graph.vec_of_rel(candidate.rel_id),
                self.graph.vec_of_ent(candidate.to_id)
            )))
        candidates_emb = np.array(emb_list)

        # 计算点积
        choices = tf.keras.layers.dot(
            [tf.expand_dims(feature, axis=0), np.expand_dims(candidates_emb, axis=0)],
            axes=2
        )

        # 计算选择概率
        probabilities = tf.keras.activations.softmax(choices[0])

        if True in np.isnan(probabilities):
            print('bad result!')

        return candidates, probabilities[0], history_state

    # prior/posterior, 通过relation区分
    # 还没有测试过，很可能有bug
    def paths_between(self, from_id, to_id, width=5, relation=None):
        step = 0
        finished = []
        states = [self.initial_state(from_id)]

        # 最大搜索history_depth跳
        while step < self.history_depth:
            all_candidates = []
            action_probs = []
            flatten_probs = np.zeros(0)
            history_states = []
            chosen_to_pos = []
            updated_states = []

            counter = 0
            for index, state in enumerate(states):
                candidates, candidate_probs, history_state = self.available_action_probs(state, relation)
                if len(candidates) == 0:
                    continue

                all_candidates.append(candidates)
                history_states.append(history_state)

                action_probs.append(candidate_probs)
                flatten_probs = np.concatenate((flatten_probs, candidate_probs))
                for action_index in range(len(candidates)):
                    chosen_to_pos.append((counter, action_index))

                counter += 1

            picks = width - len(finished)
            actions = pick_top_n(flatten_probs, picks)

            # 根据选取的动作更新路径状态
            for action_chosen in actions:
                state_index = chosen_to_pos[action_chosen][0]
                candidate_index = chosen_to_pos[action_chosen][1]
                new_state = FinderState(
                    path_step=all_candidates[state_index][candidate_index].to_list(),
                    history_state=history_states[state_index],
                    action_prob=action_probs[state_index],
                    action_chosen=candidate_index,
                    pre_state=states[state_index]
                )
                # 跳过到达目的地的路径
                if new_state.path[-1] == to_id:
                    finished.append(new_state)
                else:
                    updated_states.append(new_state)

                if len(finished) == width:
                    return finished

            states = updated_states
            step = step + 1

        return finished + states

    def learn_from_teacher(self, path, reward, rel_emb=None):
        probs = []
        gradients = []

        student_state = self.initial_state(path[0])
        for step_index in range(1, len(path), 2):
            with tf.GradientTape() as tape:
                candidates, student_action_probs, history_state \
                    = self.available_action_probs(student_state, rel_emb)

                action_index = index_of(candidates, path[step_index], path[step_index + 1])

                neg_log_prob = loss.one_hot(action_index, student_action_probs, reward)

            gradients.append(tape.gradient(neg_log_prob, self.trainable_variables))
            student_state = FinderState(
                path_step=candidates[action_index].to_list(),
                history_state=history_state,
                action_prob=student_action_probs,
                action_chosen=action_index,
                pre_state=student_state
            )
            probs.append(student_action_probs[action_index])

        return probs, gradients


if __name__ == "__main__":
    array = np.arange(5)
    print('first: {}, last: {}'.format(str(array[0]), str(array[-1])))
