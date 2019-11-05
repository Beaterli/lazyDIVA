import random
import json
from time import time
from concurrent.futures import ProcessPoolExecutor
from graph.graph import Graph
from pathfinder.brute.bfsfinder import BFSFinder

task = 'concept:athletehomestadium'
episodes_json = '{}.json'.format(task.replace(':', '_'))
db_name = 'graph.db'
search_workers = 2
max_path_length = 5
teacher_path_count = 5


def load_previous_episodes():
    try:
        json_file = open(episodes_json, 'r')
        lines = json_file.readlines()
        json_file.close()
        return json.loads('\n'.join(lines))
    except OSError:
        return None


def save_episodes(episodes):
    json_file = open(episodes_json, 'w')
    json_file.write(json.dumps(episodes))
    json_file.close()


def search(samples):
    search_result = []
    graph = Graph(db_name)
    graph.prohibit_relation(task)
    finder = BFSFinder(graph, max_path_length)
    for sample in samples:
        start_time = time()
        sample['paths'] = finder.paths_between(
            from_id=sample['from_id'],
            to_id=sample['to_id'],
            width=teacher_path_count
        )
        sample['paths'] = list(map(lambda s: s.path, sample['paths']))
        search_result.append(sample)
        duration = time() - start_time
        if duration > 5:
            print('episode: {} -> {} takes {:.2f}s!'.format(sample['from_id'], sample["to_id"], duration))
    return search_result


def all_episodes():
    return episodes


def find_episode(from_id, to_id):
    for episode in episodes:
        if episode['from_id'] == from_id \
                and episode['to_id'] == to_id:
            return episode
    return None


if __name__ == '__main__':
    graph = Graph(db_name)
    graph.prohibit_relation(task)
    previous_episodes = load_previous_episodes()

    negative_samples = graph.negative_train_samples_of(task)
    positive_samples = graph.samples_of(task, 'train', '+')
    teacher_samples = positive_samples + negative_samples
    # teacher_samples = teacher_samples[:5]
    print('using {} samples'.format(len(teacher_samples)))

    episodes = []
    thread_pool = ProcessPoolExecutor(max_workers=search_workers)
    slice_size = int(len(teacher_samples) / search_workers)
    futures = []
    search_start = time()
    for i in range(search_workers - 1):
        futures.append(thread_pool.submit(search, teacher_samples[i * slice_size:(i + 1) * slice_size]))
    futures.append(thread_pool.submit(search, teacher_samples[(search_workers - 1) * slice_size:]))

    for future in futures:
        episodes = episodes + future.result()

    thread_pool.shutdown()
    print('search complete in {}s'.format(time() - search_start))
    save_episodes(episodes)
else:
    episodes = load_previous_episodes()
