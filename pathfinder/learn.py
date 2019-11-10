import tensorflow as tf

import loss
from pathfinder.decision import index_of
from pathfinder.finderstate import FinderState


def step_by_step(finder, path, reward, rel_emb=None):
    probs = []
    gradients = []

    student_state = finder.initial_state(path[0])
    for step_index in range(1, len(path), 2):
        with tf.GradientTape() as tape:
            candidates, student_action_probs, history_state \
                = finder.available_action_probs(student_state, rel_emb)

            action_index = index_of(candidates, path[step_index], path[step_index + 1])

            neg_log_prob = loss.one_hot(action_index, student_action_probs, reward)

        gradients.append(tape.gradient(neg_log_prob, finder.trainable_variables))
        student_state = FinderState(
            path_step=candidates[action_index].to_list(),
            history_state=history_state,
            action_prob=student_action_probs,
            action_chosen=action_index,
            pre_state=student_state
        )
        probs.append(student_action_probs[action_index])

    return probs, gradients
