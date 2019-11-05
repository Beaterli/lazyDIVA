import numpy as np


def pick_top_n(probs, top_n, precision=4):
    action_space = np.arange(len(probs))

    if len(probs) <= top_n:
        return action_space

    np.round_(probs, precision)
    # 从动作空间中随机选取n个动wen作
    normalized_probs = probs / probs.sum()

    action_chosen = np.random.choice(action_space,
                                     size=top_n,
                                     replace=False,
                                     p=normalized_probs)

    return action_chosen
