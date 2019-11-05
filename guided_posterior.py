from __future__ import absolute_import, division, print_function

import time
import random

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

from graph.graph import Graph
from pathfinder.lstmfinder import LSTMFinder
import episodes

teacher_epoch = 0
teacher_path_count = 5
max_path_length = 5
task = 'concept:athletehomestadium'
db_name = 'graph.db'
emb_size = 100


def learn_episode(episode):
    if episode['type'] == '+':
        rel_emb = positive_emb
    else:
        rel_emb = negative_emb

    all_probs = []

    for path in episode['paths']:
        probs, gradients = student.learn_from_teacher(
            path=path,
            reward=1.0,
            rel_emb=rel_emb
        )

        for gradient in gradients:
            optimizer.apply_gradients(zip(gradient, student.trainable_variables))

        all_probs = all_probs + probs

    return all_probs


def learn_epoch(round, episodes):
    start_time = time.time()
    probs = []
    for index, episode in enumerate(episodes):
        probs = probs + learn_episode(episode)

    np_probs = np.array(probs)
    print('epoch: {} finished in {:.2f} seconds, prob stats: avg: {:.4f}'.format(
        i + 1,
        time.time() - start_time,
        np.average(np_probs)
    ))

    if round % 2 == 0:
        checkpoint.save(chkpt_file)


if __name__ == '__main__':
    graph = Graph(db_name)
    graph.prohibit_relation(task)
    positive_emb = graph.vec_of_rel_name(task)
    negative_emb = np.zeros(emb_size, dtype='f4')

    student = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length)
    optimizer = tf.optimizers.Adam(1e-3)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=student)

    chkpt_file = 'checkpoints/guided_posterior/posterior'

    print('eager mode: {}'.format(tf.executing_eagerly()))

    samples = episodes.all_episodes()
    random.shuffle(samples)
    print('guided learning started')
    for i in range(teacher_epoch):
        learn_epoch(i, samples)

    print('pre-train finished!')
