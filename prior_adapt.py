from __future__ import absolute_import, division, print_function

import sys
import time

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import checkpoints as chk
import episodes as eps
import loss
from graph.graph import Graph
from pathfinder.brute.bfsfinder import BFSFinder
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from train_tools import train_finder, train_reasoner, teach_finder, rollout_sample, calc_reward, show_type_distribution

epoch = 30
emb_size = 100
rollouts = 15
max_path_length = 5
samples_count = 300
save_checkpoint = True

database = sys.argv[1]
task = sys.argv[2]
task_dir_name = task.replace('/', '_').replace(':', '_')
reasoner_class = sys.argv[3]

graph = Graph(database + '.db')
graph.prohibit_relation(task)
rel_embs = {
    '+': graph.vec_of_rel_name(task),
    '-': np.zeros(emb_size, dtype='f4')
}

restore_dir = 'checkpoints/{}/{}/unified/{}/'.format(
    database,
    task_dir_name,
    reasoner_class)
checkpoint_dir = 'checkpoints/{}/{}/adapt/{}/'.format(
    database,
    task_dir_name,
    reasoner_class)

teacher = BFSFinder(env_graph=graph, max_path_length=max_path_length)
prior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=True)

if reasoner_class == 'cnn':
    path_reasoner = CNNReasoner(graph=graph, emb_size=emb_size, max_path_length=max_path_length)
else:
    path_reasoner = GraphSAGEReasoner(graph=graph, emb_size=emb_size, neighbors=15)

path_reasoner_name = type(path_reasoner).__name__
print('using {}, {}'.format(path_reasoner_name, type(prior).__name__))

likelihood_optimizer = tf.optimizers.Adam(1e-3)
# 使用SGD避免训练失败
prior_optimizer = tf.optimizers.SGD(1e-2)

likelihood_chkpt_file = checkpoint_dir + path_reasoner_name
prior_chkpt_file = checkpoint_dir + 'prior'

prior_checkpoint = tf.train.Checkpoint(optimizer=prior_optimizer, model=prior)
chk.load_latest_if_exists(
    prior_checkpoint,
    restore_dir, 'prior'
)

likelihood_checkpoint = tf.train.Checkpoint(optimizer=likelihood_optimizer, model=path_reasoner)
chk.load_latest_if_exists(
    likelihood_checkpoint,
    restore_dir, path_reasoner_name
)


# 训练likelihood
def train_likelihood(paths, label):
    return train_reasoner(
        reasoner=path_reasoner,
        optimizer=likelihood_optimizer,
        paths=paths,
        label=label
    )


# 训练prior
def train_prior(episodes):
    return train_finder(
        finder=prior,
        optimizer=prior_optimizer,
        episodes=episodes
    )


train_samples = eps.load_previous_episodes('{}.json'.format(task.replace(':', '_').replace('/', '_')))
# random.shuffle(train_samples)
train_samples = train_samples[:samples_count]
print('using {} train samples'.format(len(train_samples)))
show_type_distribution(train_samples)
# train_samples = [{
#     'from_id': 37036,
#     'to_id': 68461,
#     'type': '-'
# }]

for i in range(0, epoch * 3):
    epoch_start = time.time()
    all_loss = []
    stage = i % 3
    teacher_rounds = 0

    for index, sample in enumerate(train_samples):
        label = loss.type_to_label(sample['type'])
        rel_emb = rel_embs[sample['type']]

        paths = rollout_sample(
            finder=prior,
            sample=sample,
            rollouts=rollouts,
            rel_emb=rel_emb
        )

        positive, negative, losses = calc_reward(
            reasoner=path_reasoner,
            sample=sample,
            paths=paths,
            label=label
        )
        all_loss = all_loss + losses

        # 训练prior
        if stage == 0:
            train_prior(positive + negative)
            # 成功路径过少，需要重新监督学习
            if len(positive) < 2:
                teach_finder(
                    finder=prior,
                    optimizer=prior_optimizer,
                    sample=sample
                )
                teacher_rounds += 1
        # 训练likelihood
        else:
            if len(positive) == 0:
                continue
            paths = list(map(lambda r: r[1], positive))
            train_likelihood(paths, label)

    all_loss = np.array(all_loss)
    avg_loss = np.average(all_loss)
    min_loss = np.min(all_loss)
    print('epoch: {} stage: {} takes {:.2f}s, min: {:.4f}, avg: {:.4f}, max: {:.4f}, teaches: {}'.format(
        int(i / 2) + 1,
        stage,
        time.time() - epoch_start,
        np.min(all_loss),
        np.average(all_loss),
        np.max(all_loss),
        teacher_rounds
    ))

    if not save_checkpoint:
        continue

    wave = int(i / 2) + 1
    if wave % 5 == 0 and stage == 1:
        likelihood_checkpoint.save(likelihood_chkpt_file)
        prior_checkpoint.save(prior_chkpt_file)

print('finished!')
