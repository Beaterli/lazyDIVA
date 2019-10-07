from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

tfe = tf.contrib.eager
tf.enable_eager_execution()

import time
import numpy as np
from pathreasoner.cnn_reasoner import CNNReasoner
from pathfinder.lstmfinder import LSTMFinder
from graph.graph import Graph

epochs = 5
path_width = 400
task = 'concept_musicsong_jordan'
graph = Graph('graph.db')
graph.prohibit_relation(task)

train_set = []
test_set = []

path_finder = LSTMFinder(graph=graph, max_path_length=5)
path_reasoner = CNNReasoner(400, 3, 3)


# 使用正态分布计算logP与logQ，不一定对
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def reasoner_loss(relation, label):
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=relation, labels=label)
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


for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_set:
        paths = []
        for i in range(1, 20):
            paths.append(path_finder.path_between(train_x.from_node, train_x.to_node))
        if epoch % 3 == 0:
            for path in paths:


    end_time = time.time()

    if epoch % 1 == 0:
        loss = tfe.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
