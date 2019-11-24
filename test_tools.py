import numpy as np
import tensorflow as tf

from pathreasoner.learn import learn_from_paths

labels = ['entertainment', 'political']

def separate_dest(paths, to_id):
    positives = []
    negatives = []

    for path in paths:
        if path[-1] == to_id:
            positives.append(path)
        else:
            negatives.append(path)

    return positives, negatives


def predict_sample(sample, finder, beam, reasoner, rel_emb=None, check_dest=False):
    from_id = sample['from_id']
    to_id = sample['to_id']

    paths = list(map(
        lambda state: state.path,
        finder.paths_between(
            from_id=from_id,
            to_id=to_id,
            relation=rel_emb,
            width=beam
        ))
    )

    if check_dest:
        positives, negatives = separate_dest(paths, to_id)
    else:
        positives = paths

    if len(positives) == 0:
        print('failed! use zeros')
        return np.zeros(2, dtype='f4')

    labels = []
    for positive in positives:
        probs = reasoner.relation_of_path(positive)
        labels.append(probs)

    return tf.reduce_mean(tf.stack(labels), axis=0), positives


def predict_to_label(predict):
    top_1 = tf.argmax(predict, 0)
    return labels[top_1]


def loss_on_sample(sample, finder, beam, reasoner, label, rel_emb=None):
    from_id = sample['from_id']
    to_id = sample['to_id']

    paths = list(map(
        lambda state: state.path,
        finder.paths_between(
            from_id=from_id,
            to_id=to_id,
            relation=rel_emb,
            width=beam
        ))
    )

    positives, negatives = separate_dest(paths, to_id)
    bads = len(negatives)

    if len(positives) > 0:
        loss, gradient = learn_from_paths(
            reasoner=reasoner,
            paths=positives,
            label=label
        )
    else:
        loss = None

    return loss, bads
