from __future__ import absolute_import, division, print_function

import sys
import time

import numpy as np
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import checkpoints as chk
import episodes as eps
import loss_tools
from graph.graph import Graph
from pathfinder.brute.bfsfinder import BFSFinder
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.cnn_reasoner import CNNReasoner
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from test_tools import loss_on_sample
from train_tools import train_finder, train_reasoner, teach_finder, rollout_sample, calc_reward, show_type_distribution, \
    even_types

epoch = 25
emb_size = 100
rollouts = 5
max_path_length = 5
samples_count = 300
checkpoint_index = 5
save_checkpoint = True

database = 'weibo'
task = 'event_type'
task_dir_name = task.replace('/', '_').replace(':', '_')
reasoner_class = sys.argv[1]

graph = Graph(database + '.db')
graph.prohibit_relation('entertainment')
graph.prohibit_relation('political')
rel_embs = {
    10: graph.vec_of_rel_name('entertainment'),
    12: graph.vec_of_rel_name('political')
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

prior_checkpoint = tf.train.Checkpoint(model=prior)
chk.load_from_index(
    prior_checkpoint,
    restore_dir, 'prior',
    checkpoint_index
)

likelihood_checkpoint = tf.train.Checkpoint(model=path_reasoner)
chk.load_from_index(
    likelihood_checkpoint,
    restore_dir, path_reasoner_name,
    checkpoint_index
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


all_train_samples = eps.load_previous_episodes('{}.json'.format(task.replace(':', '_').replace('/', '_')))
# random.shuffle(train_samples)
train_samples = all_train_samples[:samples_count]
print('using {} train samples'.format(len(train_samples)))
show_type_distribution(train_samples)
# train_samples = [{
#     'from_id': 37036,
#     'to_id': 68461,
#     'type': '-'
# }]
test_index = samples_count + 1
test_count = int(samples_count / 4)
test_samples = all_train_samples[test_index:test_index + test_count]
show_type_distribution(test_samples)

for i in range(0, epoch * 2):
    epoch_start = time.time()
    all_loss = []
    stage = i % 2
    teacher_rounds = 0

    for index, sample in enumerate(train_samples):
        label = loss_tools.type_to_one_hot(sample['rid'])

        paths = rollout_sample(
            finder=prior,
            sample=sample,
            rollouts=rollouts
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

    wave = int(i / 2) + 1

    if wave % 3 == 0 and stage == 1:
        all_bads = []
        all_losses = []
        for sample in test_samples:
            label = loss_tools.type_to_one_hot(sample['rid'])

            loss, bads = loss_on_sample(
                sample=sample,
                finder=prior,
                beam=5,
                reasoner=path_reasoner,
                label=label
            )
            all_bads.append(bads)
            if loss is not None:
                all_losses.append(loss)
        print('perf on test: avg bads: {:.2f}, avg loss: {:.4f}'.format(
            np.average(np.array(all_bads)),
            np.average(np.array(all_losses))
        ))

    if not save_checkpoint:
        continue

    if wave % 5 == 0 and stage == 1:
        likelihood_checkpoint.save(likelihood_chkpt_file)
        prior_checkpoint.save(prior_chkpt_file)

print('finished!')
