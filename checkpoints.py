import os
import re


def latest_checkpoint_file(path, name):
    files = os.listdir(path)

    index_pattern = name + '-(\\d+)'
    max_index = -1
    for file in files:
        result = re.match(index_pattern, file)
        if result is not None:
            index = int(result.group(1))
            if index > max_index:
                max_index = index

    if max_index == -1:
        return None
    else:
        return name + '-' + str(max_index)


def load_latest_with_priority(
        checkpoint,
        first_path, first_name,
        second_path, second_name
):
    first = latest_checkpoint_file(first_path, first_name)
    if first is not None:
        checkpoint.restore(first_path + first)
        return
    second = latest_checkpoint_file(second_path, second_name)
    if second is not None:
        checkpoint.restore(second_path + second)


def load_latest_if_exists(checkpoint, path, name):
    file = latest_checkpoint_file(path, name)
    if file is not None:
        checkpoint.restore(path + file)


if __name__ == '__main__':
    print(latest_checkpoint_file('checkpoints/guided_posterior', 'posterior'))
    print(latest_checkpoint_file('checkpoints/guided_posterior', 'prior'))
