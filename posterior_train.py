from __future__ import absolute_import, division, print_function

import time

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import loss
from graph.graph import Graph
from pathfinder.finderstate import FinderState
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner

epoch = 10
task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)
checkpoint_dir = 'checkpoints/'

train_set = []
test_set = []

path_finder = LSTMFinder(graph=graph, emb_size=100, max_path_length=5, prior=False)
path_finder.load_weights(checkpoint_dir + 'student')
path_reasoner = CNNReasoner(graph=graph, input_width=200, max_path_length=5)
# 最终奖励值倒推时的递减系数
reward_param = 0.8

emb_size = 100
rollouts = 20

optimizer = tf.optimizers.Adam(1e-4)


def reason_loss(relation, expected_label):
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=relation, labels=expected_label)
    # 分类结果熵向量求和
    return tf.reduce_sum(cross_ent, axis=[0])


# train_samples = graph.samples_of(task, "train", "+")[1:100]
train_samples = graph.train_samples_of(task)[1:100]
# train_samples = [{
#     'from_id': 71675,
#     'to_id': 3749,
#     'type': '+'
# }]
test_samples = graph.test_samples_of(task)
rel_emb = graph.vec_of_rel_name(task)
for i in range(epoch):
    print('epoch: {} started!, samples: {}'.format(i, len(train_samples)))
    for index, episode in enumerate(train_samples):
        print('episode: {} started!'.format(episode))
        rewards = []

        start_time = time.time()

        label = loss.type_to_label(episode['type'])

        # 查找n条路径
        path_states = path_finder.paths_between(episode['from_id'], episode['to_id'], rollouts, rel_emb)

        # 训练likelihood, 计算奖励
        for state in path_states:
            if not state.path[-1] == episode['to_id']:
                rewards.append(-1)
                continue

            with tf.GradientTape() as likelihood_tape:
                classify_loss = reason_loss(path_reasoner.relation_of_path(state.path), label)

            gradient = likelihood_tape.gradient(classify_loss, path_reasoner.trainable_variables)
            optimizer.apply_gradients(zip(gradient, path_reasoner.trainable_variables))
            rewards.append(classify_loss)

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
                    path_step=candidates[taken_action].to_tuple(),
                    history_state=history_state,
                    action_prob=action_probs,
                    action_chosen=taken_action,
                    pre_state=playback_state
                )
                gradient = posterior_tape.gradient(neg_log_prob, path_finder.trainable_variables)
                optimizer.apply_gradients(zip(gradient, path_finder.trainable_variables))
                # print('updated with reward: {}'.format(reward))

        end_time = time.time()
        good_rewards = np.array(list(filter(lambda r: r > 0.0, rewards)))
        if len(good_rewards) == 0:
            avg_reward = -1
        else:
            avg_reward = np.average(good_rewards)
        print('time for an episode: {:.2f}s'.format(end_time - start_time))
        print('bad paths: {}, avg good loss: {}'.format(
            rewards.count(-1),
            avg_reward
        ))

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

print('finished!')
