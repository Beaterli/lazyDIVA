import random

from pathfinder.learn import step_by_step
from pathreasoner.learn import learn_from_path, learn_from_paths

teacher_reward = 1.0
search_failure_reward = -0.05
success_reward_range = [0.0, 0.5]


def clip_range(value, min_value, max_value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def show_type_distribution(samples):
    dist = {}
    for sample in samples:
        sample_type = sample['type']
        if sample_type not in dist:
            dist[sample_type] = 0
        dist[sample_type] = dist[sample_type] + 1
    print('type distribution: {}'.format(str(dist)))


def even_types(samples, count):
    positives = []
    negatives = []
    for sample in samples:
        if sample['type'] == "+":
            positives.append(sample)
        else:
            negatives.append(sample)

    random.shuffle(positives)
    random.shuffle(negatives)

    total = positives[:int(count / 2)] + negatives[:int(count / 2)]
    random.shuffle(total)
    return total


# 训练path finder
def train_finder(finder, optimizer, episodes, rel_emb=None):
    all_probs = []
    for reward, path in episodes:
        if reward < 0.05:
            continue

        probs, gradients = step_by_step(
            finder=finder,
            path=path,
            reward=reward,
            rel_emb=rel_emb
        )
        all_probs = all_probs + probs

        for gradient in gradients:
            optimizer.apply_gradients(zip(gradient, finder.trainable_variables))

    return all_probs


# 搜索失败时重新训练
def teach_finder(finder, optimizer, sample, rel_emb=None, teacher=None):
    if teacher is not None:
        states = teacher.paths_between(sample['from_id'], sample['to_id'], 5)
        paths = list(map(lambda s: s.path, states))
    else:
        paths = sample['paths']

    episodes = list(map(lambda p: (teacher_reward, p), paths))

    return train_finder(finder, optimizer, episodes, rel_emb)


# 训练likelihood
def train_reasoner(reasoner, optimizer, paths, label):
    classify_loss, gradient = learn_from_paths(
        reasoner=reasoner,
        paths=paths,
        label=label
    )
    optimizer.apply_gradients(zip(gradient, reasoner.trainable_variables))


# 查找n条路径
def rollout_sample(finder, sample, rollouts, rel_emb=None):
    path_states = finder.paths_between(sample['from_id'], sample['to_id'], rollouts, rel_emb)

    return map(lambda s: s.path, path_states)


# 计算奖励值
def calc_reward(reasoner, sample, paths, label):
    positive_results = []
    negative_results = []
    losses = []

    # 获得路径的奖励值
    for path in paths:
        # if path[-1] != sample['to_id']:
        #     negative_results.append((search_failure_reward, path))
        #     continue

        # 需要反转分类损失作为路径搜索奖励
        classify_loss, gradient = learn_from_path(reasoner, path, label)
        losses.append(classify_loss)
        reward = clip_range(0.8 - classify_loss, success_reward_range[0], success_reward_range[1])
        positive_results.append((reward, path))

    return positive_results, negative_results, losses
