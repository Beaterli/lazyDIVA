from __future__ import absolute_import, division, print_function

import random
import time

import numpy as np
import episodes as eps
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import loss
from graph.graph import Graph
from pathfinder.brute.bfsfinder import BFSFinder
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner

epoch = 25
emb_size = 100
rollouts = 5
max_path_length = 5

task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)
checkpoint_dir = 'checkpoints/'

train_set = []
test_set = []

teacher = BFSFinder(env_graph=graph, max_path_length=max_path_length)
path_finder = LSTMFinder(graph=graph, emb_size=100, max_path_length=max_path_length, prior=False)
path_reasoner = CNNReasoner(graph=graph, input_width=200, max_path_length=max_path_length)

likelihood_optimizer = tf.optimizers.Adam(1e-4)
posterior_optimizer = tf.optimizers.Adam(1e-3)

posterior_checkpoint = tf.train.Checkpoint(optimizer=posterior_optimizer, model=path_finder)
posterior_chkpt_file = 'checkpoints/posterior_fine'

likelihood_checkpoint = tf.train.Checkpoint(optimizer=likelihood_optimizer, model=path_reasoner)
likelihood_chkpt_file = 'checkpoints/likelihood'

train_samples = eps.all_episodes()
random.shuffle(train_samples)
test_samples = graph.test_samples_of(task)
# train_samples = [{
#     'from_id': 37036,
#     'to_id': 68461,
#     'type': '-'
# }]
positive_rel_emb = graph.vec_of_rel_name(task)
negative_rel_emb = np.zeros(emb_size, dtype='f4')
search_failure_reward = -0.05

posterior_checkpoint.restore('checkpoints/posterior-25')


def train_episode(episode):
    positive_results = []
    negative_results = []
    all_probs = []
    start_time = time.time()

    label = loss.type_to_label(episode['type'])
    if episode['type'] == '+':
        rel_emb = positive_rel_emb
    else:
        rel_emb = negative_rel_emb

    # 查找n条路径
    path_states = path_finder.paths_between(episode['from_id'], episode['to_id'], rollouts, rel_emb)

    # 训练likelihood, 计算奖励
    for state in path_states:
        if state.path[-1] != episode['to_id']:
            negative_results.append((search_failure_reward, state.path))
            continue

        # 分类损失为0.0-1.0
        classify_loss, gradient = path_reasoner.learn_from_label(state.path, label)
        likelihood_optimizer.apply_gradients(zip(gradient, path_reasoner.trainable_variables))

        # 需要反转分类损失作为路径搜索奖励
        positive_results.append((1.0 - classify_loss, state.path))

    # 训练posterior, 成功路径
    for reward, path in positive_results:
        probs, gradients = path_finder.learn_from_teacher(
            path,
            rel_emb,
            reward
        )
        all_probs = all_probs + probs

        for gradient in gradients:
            posterior_optimizer.apply_gradients(zip(gradient, path_finder.trainable_variables))

    # 训练posterior, 失败路径
    for reward, path in negative_results:
        probs, gradients = path_finder.learn_from_teacher(
            path,
            rel_emb,
            reward
        )
        all_probs = all_probs + probs

        for gradient in gradients:
            posterior_optimizer.apply_gradients(zip(gradient, path_finder.trainable_variables))

    # 成功路径过少，需要重新监督学习
    if len(positive_results) < 3:
        print('reteaching posterior')
        paths = eps.find_episode(episode['from_id'], episode['to_id'])['paths']
        if paths is None:
            states = teacher.paths_between(episode['from_id'], episode['to_id'], 5)
            paths = list(map(lambda s: s.path, states))

        for path in paths:
            probs, gradients = path_finder.learn_from_teacher(
                path,
                rel_emb,
                1.0
            )
            for gradient in gradients:
                posterior_optimizer.apply_gradients(zip(gradient, path_finder.trainable_variables))

    end_time = time.time()
    good_loss = np.array(list(map(lambda r: 1.0 - r[0], positive_results)))
    if len(good_loss) == 0:
        avg_loss = -1
    else:
        avg_loss = np.average(good_loss)
    print('time for {} episode: {:.2f}s, bad paths: {}, avg good loss: {:.4f}'
          .format(episode['type'],
                  end_time - start_time,
                  len(negative_results),
                  avg_loss))


for i in range(epoch):
    print('epoch: {} started!, samples: {}'.format(i, len(train_samples)))
    for index, sample in enumerate(train_samples):
        train_episode(sample)

    posterior_checkpoint.save(posterior_chkpt_file)
    likelihood_checkpoint.save(likelihood_chkpt_file)

print('finished!')
