from __future__ import absolute_import, division, print_function

import random
import time

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import checkpoints as chk
import episodes as eps
import loss
from graph.graph import Graph
from pathfinder.brute.bfsfinder import BFSFinder
from pathfinder.learn import learn_from_teacher
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from pathreasoner.learn import learn_from_paths, learn_from_path

epoch = 25
emb_size = 100
rollouts = 5
max_path_length = 5
samples_count = 5

task = 'concept:athletehomestadium'
graph = Graph('graph.db')
graph.prohibit_relation(task)
checkpoint_dir = 'checkpoints/'

train_set = []
test_set = []

teacher = BFSFinder(env_graph=graph, max_path_length=max_path_length)
posterior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=False)
prior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=True)
# path_reasoner = CNNReasoner(graph=graph, emb_size=emb_size, max_path_length=max_path_length)
path_reasoner = GraphSAGEReasoner(graph=graph, emb_size=emb_size, neighbors=25)

likelihood_optimizer = tf.optimizers.Adam(3e-5)
# 使用SGD避免训练失败
posterior_optimizer = tf.optimizers.SGD(1e-2)
# 使用Adam提升学习速度
prior_optimizer = tf.optimizers.Adam(2e-3)

save_checkpoint = False

posterior_chkpt_file = 'checkpoints/unified/posterior_fine'
posterior_checkpoint = tf.train.Checkpoint(model=posterior)
chk.load_latest_with_priority(
    posterior_checkpoint,
    'checkpoints/unified/', 'posterior_fine',
    'checkpoints/guided_posterior/', 'posterior'
)

prior_checkpoint = tf.train.Checkpoint(optimizer=prior_optimizer, model=prior)
prior_chkpt_file = 'checkpoints/unified/prior'
chk.load_latest_if_exists(
    prior_checkpoint,
    'checkpoints/unified/', 'prior'
)

likelihood_checkpoint = tf.train.Checkpoint(optimizer=likelihood_optimizer, model=path_reasoner)
likelihood_chkpt_file = 'checkpoints/unified/likelihood'
chk.load_latest_if_exists(
    likelihood_checkpoint,
    'checkpoints/unified/', 'likelihood'
)


def train_finder(finder, episodes, rel_emb):
    all_probs = []
    for reward, path in episodes:
        probs, gradients = learn_from_teacher(
            finder=finder,
            path=path,
            reward=reward,
            rel_emb=rel_emb
        )
        all_probs = all_probs + probs

        for gradient in gradients:
            posterior_optimizer.apply_gradients(zip(gradient, finder.trainable_variables))

    return all_probs


# 搜索失败时重新训练
def teach_posterior(from_id, to_id):
    episode = eps.find_episode(from_id, to_id)

    if episode['type'] == '+':
        rel_emb = positive_rel_emb
    else:
        rel_emb = negative_rel_emb

    paths = episode['paths']
    if paths is None:
        states = teacher.paths_between(sample['from_id'], sample['to_id'], 5)
        paths = list(map(lambda s: s.path, states))
    paths = list(map(lambda p: (1.0, p), paths))

    train_finder(posterior, paths, rel_emb)


# 训练posterior
def train_posterior(positive, negative, rel_emb):
    return train_finder(posterior, positive + negative, rel_emb)


# 训练likelihood
def train_likelihood(paths, label):
    classify_loss, gradient = learn_from_paths(
        reasoner=path_reasoner,
        paths=paths,
        label=label
    )
    likelihood_optimizer.apply_gradients(zip(gradient, path_reasoner.trainable_variables))


# 训练prior
def train_prior(results):
    return train_finder(prior, results, None)


def rollout_episode(episode, rel_emb, label):
    positive_results = []
    negative_results = []

    # 查找n条路径
    path_states = posterior.paths_between(episode['from_id'], episode['to_id'], rollouts, rel_emb)

    # 获得路径的奖励值
    for state in path_states:
        if state.path[-1] != episode['to_id']:
            negative_results.append((search_failure_reward, state.path))
            continue

        # 需要反转分类损失作为路径搜索奖励
        classify_loss, gradient = learn_from_path(path_reasoner, state.path, label)
        positive_results.append((1.0 - classify_loss, state.path))

    return positive_results, negative_results


train_samples = eps.all_episodes()
random.shuffle(train_samples)
train_samples = train_samples[:samples_count]
print('using {} train samples'.format(len(train_samples)))

# train_samples = [{
#     'from_id': 37036,
#     'to_id': 68461,
#     'type': '-'
# }]
positive_rel_emb = graph.vec_of_rel_name(task)
negative_rel_emb = np.zeros(emb_size, dtype='f4')
search_failure_reward = -0.05

for i in range(epoch * 3):
    epoch_start = time.time()
    all_loss = np.zeros(0)
    stage = i % 3
    teacher_rounds = 0

    for index, sample in enumerate(train_samples):
        label = loss.type_to_label(sample['type'])
        if sample['type'] == '+':
            rel_emb = positive_rel_emb
        else:
            rel_emb = negative_rel_emb

        positive, negative = rollout_episode(sample, rel_emb, label)
        all_loss = np.concatenate((all_loss, list(map(lambda r: max(1.0 - r[0], 0.05), positive))))

        # 训练posterior
        if stage == 0:
            train_posterior(positive, negative, rel_emb)
            # 成功路径过少，需要重新监督学习
            if len(positive) < 2:
                teach_posterior(sample['from_id'], sample['to_id'])
                teacher_rounds += 1
        # 训练likelihood
        elif stage == 1:
            if len(positive) == 0:
                continue
            paths = list(map(lambda r: r[1], positive))
            train_likelihood(paths, label)
        # 训练prior
        else:
            train_prior(positive)

    avg_loss = np.average(all_loss)
    min_loss = np.min(all_loss)
    print('epoch: {} stage: {} takes {:.2f}s, min: {:.4f}, avg: {:.4f}, max: {:.4f}, teaches: {}'.format(
        int(i / 3) + 1,
        stage,
        time.time() - epoch_start,
        np.min(all_loss),
        np.average(all_loss),
        np.max(all_loss),
        teacher_rounds
    ))

    if not save_checkpoint:
        continue

    if i % 15 == 12:
        posterior_checkpoint.save(posterior_chkpt_file)
    elif i % 15 == 13:
        likelihood_checkpoint.save(likelihood_chkpt_file)
    elif i % 15 == 14:
        prior_checkpoint.save(prior_chkpt_file)

if save_checkpoint:
    posterior_checkpoint.save(posterior_chkpt_file)
    likelihood_checkpoint.save(likelihood_chkpt_file)
    prior_checkpoint.save(prior_chkpt_file)
print('finished!')
