from __future__ import absolute_import, division, print_function

import time

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

from graph.graph import Graph
from pathfinder.bfsfinder import BFSFinder
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner

tfe = tf.contrib.eager
tf.enable_eager_execution()

teacher_epoch = 10
teacher_path_count = 3
epoch = 10
task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)

train_set = []
test_set = []

teacher = BFSFinder(graph=graph, max_path_length=5)
path_finder = LSTMFinder(graph=graph, emb_size=100, max_path_length=5)
path_reasoner = CNNReasoner(graph=graph, input_width=200, max_path_length=5)
variables = path_finder.trainable_variables + path_reasoner.trainable_variables
# 最终奖励值倒推时的递减系数
reward_param = 0.8

emb_size = 100


# 使用正态分布计算logP与logQ，不一定对
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def reasoner_loss(relation, expected_label):
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=relation, labels=expected_label)
    # 分类结果熵向量求和
    return -tf.reduce_sum(cross_ent, axis=[0])


def divergence_loss(prior_path, posterior_path):
    # prior计算损失
    log_prior = log_normal_pdf(prior_path, 0., 0.)
    # posterior计算损失
    log_posterior = log_normal_pdf(posterior_path, 0., 0.)
    return log_prior - log_posterior


optimizer = tf.train.AdamOptimizer(1e-4)


def one_hot_loss(target_action, action_prob, reward):
    action_dim = action_prob.shape[0]
    action_onehot = tf.one_hot(target_action, action_dim)
    action_mask = tf.cast(action_onehot, tf.bool)
    picked_prob = tf.boolean_mask(action_prob, action_mask)
    action_loss = tf.reduce_sum(-tf.log(picked_prob) * reward)
    return action_loss


def reasoner_update(reason_loss, tape):
    gradients = tape.gradient(reason_loss, variables)
    optimizer.apply_gradients(zip(gradients, path_reasoner.trainable_variables))


def posterior_update(posterior_loss, tape):
    gradients = tape.gradient(posterior_loss, variables)
    optimizer.apply_gradients(zip(gradients, path_finder.trainable_variables))


def mdp_reinforce_update(reason_loss, actions, action_probs, tapes):
    for step_index, step_prob in enumerate(action_probs):
        reward = one_hot_loss(actions[step_index], step_prob, reason_loss)
        gradients = tapes[step_index].gradient(reward, path_finder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, path_finder.trainable_variables))
    return


def prior_update(prior_loss, tape):
    gradients = tape.gradient(prior_loss, variables)
    optimizer.apply_gradients(zip(gradients, path_finder.trainable_variables))


teacher_samples = graph.samples_of(task, "train", "+")[1:100]
for i in range(teacher_epoch):
    print('teacher epoch: {} started!, samples: {}'.format(i, len(teacher_samples)))
    for index, episode in enumerate(teacher_samples):
        start_time = time.time()

        rel_emb = graph.vec_of_rel_name(task)
        teacher_states = teacher.paths_between(episode['from_id'], episode['to_id'], teacher_path_count)
        student_states = path_finder.paths_between(episode['from_id'], episode['to_id'], teacher_path_count, rel_emb)

        for j in range(teacher_path_count):
            teacher_actions = teacher_states[j].action_chosen
            student_actions = student_states[j].action_chosen
            student_action_probs = student_states[j].action_probs
            tapes = student_states[j].tapes

            step_count = min(len(teacher_actions), len(student_actions))
            for k in range(step_count):
                loss = one_hot_loss(teacher_actions[k], student_action_probs, 1)
                gradient = tapes[k].gradient(loss, path_finder.trainable_variables)
                optimizer.apply_gradients(zip(gradient, path_finder.trainable_variables))

        end_time = time.time()
        print('time for episode: {} is {}'.format(episode, end_time - start_time))

train_samples = graph.train_samples_of(task)[1:100]
test_samples = graph.test_samples_of(task)
for i in range(epoch):
    print('epoch: {} started!, samples: {}'.format(i, len(train_samples)))
    for index, episode in enumerate(train_samples):
        start_time = time.time()
        label = np.array([0.0, 1.0], dtype='f4')
        rel_emb = np.zeros(emb_size)
        if episode['type'] == '+':
            rel_emb = graph.vec_of_rel_name(task)
            label = np.array([1.0, 0.0], dtype='f4')

        # 从posterior rollout K个路径
        with tf.GradientTape() as gradient_tape:
            path_states = path_finder.paths_between(episode['from_id'], episode['to_id'], 3, rel_emb)

            paths = []
            prob_stacks = []
            action_chosens = []
            for state in path_states:
                paths.append(state.path)
                prob_stacks.append(state.action_probs)
                action_chosens.append(state.action_chosen)

            print("Paths: " + str(paths))

            # Monte-Carlo REINFORCE奖励计算
            log_pr = 0.0
            for path_index, path in enumerate(paths):
                path_loss = reasoner_loss(path_reasoner.relation_of_path(path), label)
                mdp_reinforce_update(path_loss, action_chosens[path_index], prob_stacks[path_index],
                                     gradient_tape)
                log_pr += path_loss
            log_pr = log_pr / len(paths)

            # 按照posterior->likelihood->prior的顺序更新
            if index % 3 == 0:
                posterior_update(log_pr, gradient_tape)
            elif index % 3 == 1:
                reasoner_update(log_pr, gradient_tape)
        # else:
        #     prior_paths = path_finder.paths_between(episode['from_id'], episode['to_id'], 20)
        #     prior_update(divergence_loss(prior_paths, paths))

        end_time = time.time()
        print('time for an episode: {}'.format(end_time - start_time))

    # 测试每一轮的训练效果
    loss = tfe.metrics.Mean()
    for episode in test_samples:
        label = 0
        if episode['type'] == '+':
            label = 1

        # beam search 5条路径
        paths = path_finder.paths_between(episode['from_id'], episode['to_id'], 5)
        min_loss = 9999.9
        # 取误差最小的1条
        for path in paths:
            loss = -reasoner_loss(path_reasoner.relation_of_path(path), label)
            if loss < min_loss:
                min_loss = loss

        loss(min_loss)
    avg_loss = loss.result()

    print('epoch: {} finished! average loss: {}'.format(i, avg_loss))

print('finished!')
