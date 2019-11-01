import tensorflow as tf
from tensorflow import math


def one_hot(target_action, action_prob, reward):
    action_dim = action_prob.shape[0]
    action_onehot = tf.one_hot(target_action, action_dim)
    action_mask = tf.cast(action_onehot, tf.bool)
    picked_prob = tf.boolean_mask(action_prob, action_mask)
    action_loss = tf.reduce_sum(-math.log(picked_prob) * reward)
    return action_loss
