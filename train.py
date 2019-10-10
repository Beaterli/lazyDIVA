from __future__ import absolute_import, division, print_function

import time

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

from graph.graph import Graph
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner

tfe = tf.contrib.eager
tf.enable_eager_execution()

epoch = 10
task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)

train_set = []
test_set = []

path_finder = LSTMFinder(graph=graph, emb_size=100, max_path_length=5)
path_reasoner = CNNReasoner(200, 3, 3)

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
    return -tf.reduce_sum(cross_ent, axis=[1])


def divergence_loss(prior_path, posterior_path):
    # prior计算损失
    log_prior = log_normal_pdf(prior_path, 0., 0.)
    # posterior计算损失
    log_posterior = log_normal_pdf(posterior_path, 0., 0.)
    return log_prior - log_posterior


optimizer = tf.train.AdamOptimizer(1e-4)


def reasoner_update(reason_loss):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(reason_loss, path_reasoner.trainable_variables)
        optimizer.apply_gradients(zip(gradients, path_reasoner.trainable_variables))


def posterior_update(posterior_loss):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(posterior_loss, path_finder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, path_finder.trainable_variables))


def prior_update(prior_loss):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(prior_loss, path_finder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, path_finder.trainable_variables))


train_samples = graph.train_samples_of(task)
test_samples = graph.test_samples_of(task)
for i in range(epoch):
    print('epoch: {} started!'.format(i))
    for episode in train_samples:
        start_time = time.time()
        label = 0
        rel_emb = np.zeros(emb_size)
        if episode['type'] == '+':
            rel_emb = graph.vec_of_rel_name(task)
            label = 1

        # 从posterior rollout K个路径
        paths = path_finder.paths_between(episode['from_id'], episode['to_id'], rel_emb, 20)

        # Monte-Carlo REINFORCE奖励计算
        log_pr = 0.0
        for path in paths:
            log_pr += reasoner_loss(path_reasoner.relation_of_path(path), label)
        log_pr = log_pr / len(paths)

        # 按照posterior->likelihood->prior的顺序更新
        if episode % 3 == 0:
            posterior_update(log_pr)
        elif episode % 3 == 1:
            reasoner_update(log_pr)
        else:
            prior_paths = path_finder.paths_between(episode['from_id'], episode['to_id'], 20)
            prior_update(divergence_loss(prior_paths, paths))

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
