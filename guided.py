from __future__ import absolute_import, division, print_function

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

from graph.graph import Graph
from pathfinder.brute.bfsfinder import BFSFinder
from pathfinder.lstmfinder import LSTMFinder

teacher_epoch = 50
teacher_path_count = 5
max_path_length = 5
task = 'concept:athletehomestadium'
db_name = 'graph.db'
emb_size = 100
episodes_json = 'episodes.json'


def load_previous_episodes():
    try:
        json_file = open(episodes_json, 'r')
        lines = json_file.readlines()
        json_file.close()
        return json.loads('\n'.join(lines))
    except OSError:
        return None


def save_episodes(episodes):
    json_file = open(episodes_json, 'w')
    json_file.write(json.dumps(episodes))
    json_file.close()


def search(samples):
    search_result = []
    graph = Graph(db_name)
    graph.prohibit_relation(task)
    finder = BFSFinder(graph, max_path_length)
    for sample in samples:
        start_time = time.time()
        sample['states'] = finder.paths_between(
            from_id=sample['from_id'],
            to_id=sample['to_id'],
            width=teacher_path_count
        )
        search_result.append(sample)
        duration = time.time() - start_time
        if duration > 5:
            print('episode: {} -> {} takes {:.2f}s!'.format(sample['from_id'], sample["to_id"], duration))
    return search_result


def learn_episode(episode):
    if episode['type'] == '+':
        rel_emb = positive_emb
    else:
        rel_emb = negative_emb
    probs, gradients = student.learn_from_teacher(episode, rel_emb)
    for gradient in gradients:
        optimizer.apply_gradients(zip(gradient, student.trainable_variables))
    return probs


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

    chkpt_file = 'checkpoints/posterior'

    print('eager mode: {}'.format(tf.executing_eagerly()))

    previous_episodes = load_previous_episodes()
    if previous_episodes is None:
        negative_samples = graph.negative_train_samples_of(task)
        positive_samples = graph.samples_of(task, 'train', '+')
        teacher_samples = positive_samples + negative_samples
        random.shuffle(teacher_samples)
        # teacher_samples = teacher_samples[:20]
        print('using {} samples'.format(len(teacher_samples)))

        episodes = []
        workers = 8
        thread_pool = ProcessPoolExecutor(max_workers=workers)
        slice_size = int(len(teacher_samples) / workers)
        futures = []
        search_start = time.time()
        for i in range(workers - 1):
            futures.append(thread_pool.submit(search, teacher_samples[i * slice_size:(i + 1) * slice_size]))
        futures.append(thread_pool.submit(search, teacher_samples[(workers - 1) * slice_size:]))

        for future in futures:
            episodes = episodes + future.result()

        thread_pool.shutdown()
        print('search complete in {}s'.format(time.time() - search_start))
        save_episodes(episodes)
    else:
        episodes = previous_episodes

    print('guided learning started')
    for i in range(teacher_epoch):
        learn_epoch(i, episodes)

    print('pre-train finished!')
