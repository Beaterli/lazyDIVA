from __future__ import absolute_import, division, print_function

import random
import time

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import loss
from graph.graph import Graph
from pathfinder.finderstate import FinderState
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner

epoch = 25
task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)
checkpoint_dir = 'checkpoints/'

train_set = []
test_set = []

path_finder = LSTMFinder(graph=graph, emb_size=100, max_path_length=5, prior=False)
path_reasoner = CNNReasoner(graph=graph, input_width=200, max_path_length=5)
# 最终奖励值倒推时的递减系数
reward_param = 0.8

emb_size = 100
rollouts = 20

likelihood_optimizer = tf.optimizers.Adam(1e-4)
posterior_optimizer = tf.optimizers.Adam(1e-5)

posterior_checkpoint = tf.train.Checkpoint(optimizer=posterior_optimizer, model=path_finder)
posterior_chkpt_file = 'checkpoints/posterior_fine'

likelihood_checkpoint = tf.train.Checkpoint(optimizer=likelihood_optimizer, model=path_reasoner)
likelihood_chkpt_file = 'checkpoints/likelihood'

positive_samples = graph.samples_of(task, "train", "+")
negative_samples = graph.negative_train_samples_of(task)
train_samples = positive_samples + negative_samples
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
    print('episode: {} started!'.format(episode))
    rewards = []

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
            rewards.append(search_failure_reward)
            continue

        # 分类损失为0-1
        classify_loss, gradient = path_reasoner.learn_from_label(state.path, label)
        likelihood_optimizer.apply_gradients(zip(gradient, path_reasoner.trainable_variables))

        # 需要反转分类损失作为路径搜索奖励
        rewards.append(1.0 - classify_loss)

    for path_index, state in enumerate(path_states):
        reward = rewards[path_index]

        playback_state = path_finder.initial_state(episode['from_id'])

        for taken_action in state.action_chosen:
            with tf.GradientTape() as posterior_tape:
                candidates, action_probs, history_state \
                    = path_finder.available_action_probs(playback_state, rel_emb)
                neg_log_prob = loss.one_hot(taken_action, action_probs, reward)

            if taken_action > len(candidates):
                print('action outside of candidates, state: {}'.format(state))

            playback_state = FinderState(
                path_step=candidates[taken_action].to_list(),
                history_state=history_state,
                action_prob=action_probs,
                action_chosen=taken_action,
                pre_state=playback_state
            )
            gradient = posterior_tape.gradient(neg_log_prob, path_finder.trainable_variables)
            posterior_optimizer.apply_gradients(zip(gradient, path_finder.trainable_variables))
            # print('updated with reward: {}'.format(reward))

    end_time = time.time()
    good_rewards = np.array(list(filter(lambda r: r > 0.0, rewards)))
    if len(good_rewards) == 0:
        avg_reward = -1
    else:
        avg_reward = np.average(good_rewards)
    print('time for an episode: {:.2f}s'.format(end_time - start_time))
    print('bad paths: {}, avg good reward: {}'.format(
        rewards.count(search_failure_reward),
        avg_reward
    ))


for i in range(epoch):
    print('epoch: {} started!, samples: {}'.format(i, len(train_samples)))
    for index, sample in enumerate(train_samples):
        train_episode(sample)

    # # 测试每一轮的训练效果
    # metric = tf.keras.metrics.Mean()
    # for episode in test_samples:
    #     label = loss.type_to_label(episode['type'])
    #
    #     # beam search 5条路径
    #     paths = path_finder.paths_between(episode['from_id'], episode['to_id'], rollouts, rel_emb)
    #     min_loss = 9999.9
    #     # 取误差最小的1条
    #     for path in paths:
    #         loss = reason_loss(path_reasoner.relation_of_path(path), label)
    #         if loss < min_loss:
    #             min_loss = loss
    #
    #     metric(min_loss)
    # avg_loss = metric.result()
    #
    # print('epoch: {} finished! average loss: {}'.format(i, avg_loss))

    posterior_checkpoint.save(posterior_chkpt_file)
    likelihood_checkpoint.save(likelihood_chkpt_file)

print('finished!')
