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

graph = Graph()
train_set = []
test_set = []
posterior = LSTMFinder(embedding_size=200, graph=graph, max_path_length=3, mlp_size=400)
path_finder = LSTMFinder(embedding_size=200, graph=graph, max_path_length=3, mlp_size=400)
path_reasoner = CNNReasoner(400, 3)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(train_data):
    path = path_finder.path_between(train_data.from_node, train_data.to_node)
    relation = path_reasoner.relation_of_path(path)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=relation, labels=train_data.relation)
    logpr_l = -tf.reduce_sum(cross_ent, axis=[1])
    logpl = log_normal_pdf(path, 0., 0.)
    logql_r = log_normal_pdf(path, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


optimizer = tf.train.AdamOptimizer(1e-4)


def apply_gradients(optimizer, gradients, variables, global_step=None):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_set:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
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
