import tensorflow as tf
from tensorflow import math


def type_to_one_hot(episode_type):
    if episode_type == '+':
        label_index = 0
    else:
        label_index = 1
    return tf.cast(tf.one_hot(label_index, 2), tf.float32)


def type_to_label(episode_type):
    if episode_type == '+':
        return 0
    else:
        return 1


def one_hot(target_action, action_prob, reward):
    action_dim = action_prob.shape[0]
    action_onehot = tf.one_hot(target_action, action_dim)
    action_mask = tf.cast(action_onehot, tf.bool)
    picked_prob = tf.boolean_mask(action_prob, action_mask)
    action_loss = tf.reduce_sum(-math.log(picked_prob) * reward)
    return action_loss
