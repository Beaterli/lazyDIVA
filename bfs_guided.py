from __future__ import absolute_import, division, print_function

import math
import time

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import loss
from graph.graph import Graph
from pathfinder.bfsfinder import BFSFinder
from pathfinder.finderstate import FinderState
from pathfinder.lstmfinder import LSTMFinder

tfe = tf.contrib.eager
tf.enable_eager_execution()

teacher_epoch = 25
teacher_path_count = 3
max_path_length = 5
epoch = 10
task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)
rel_emb = graph.vec_of_rel_name(task)

teacher = BFSFinder(env_graph=graph, max_path_length=5)
student = LSTMFinder(graph=graph, emb_size=100, max_path_length=5)

optimizer = tf.train.AdamOptimizer(5e-4)

print(tf.config.experimental.list_physical_devices("CPU"))

teacher_samples = graph.samples_of(task, "train", "+")

quick_samples = []
quick_samples_states = []
for episode in teacher_samples:
    start_time = time.time()
    states = teacher.paths_between(
        from_id=episode['from_id'],
        to_id=episode['to_id'],
        width=teacher_path_count
    )
    if time.time() - start_time < 0.5:
        quick_samples.append(episode)
        quick_samples_states.append(states)
    else:
        print('skipped episode: {} for long search time!'.format(episode))

for i in range(teacher_epoch):
    print('teacher epoch: {} started!, samples: {}'.format(i, len(teacher_samples)))
    avg_diff = 0.0
    count = 0
    for index, episode in enumerate(quick_samples):
        start_time = time.time()

        teacher_states = quick_samples_states[index]

        # print('episode:{} takes {}s by BFS'.format(episode, time.time() - start_time))

        for teacher_state in teacher_states:
            student_state = student.initial_state(episode['from_id'])

            for label_action in teacher_state.action_chosen:
                with tf.GradientTape() as tape:
                    candidates, student_action_probs, history_state \
                        = student.available_action_probs(student_state, rel_emb)

                    updated_state = FinderState(
                        path_step=(candidates[label_action].rel_id, candidates[label_action].to_id),
                        history_state=history_state,
                        action_prob=student_action_probs,
                        action_chosen=label_action,
                        pre_state=student_state
                    )

                    diff = loss.one_hot(label_action, student_action_probs, 1)
                    gradient = tape.gradient(diff, student.trainable_variables)
                    optimizer.apply_gradients(zip(gradient, student.trainable_variables))

                    avg_diff = avg_diff + diff
                    count = count + 1

                student_state = updated_state

        end_time = time.time()
        # print('time for episode: {} is {}'.format(episode, end_time - start_time))

    print('epoch: {} finished! avg prob is {}'.format(i, math.pow(2, -1 * avg_diff / count)))

print('pre-train finished!')
