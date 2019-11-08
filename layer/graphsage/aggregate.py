import numpy as np
import tensorflow as tf


def recursive(graph, aggregator, root_id, max_depth, max_neighbor, depth=0, skip=(), emb_override={}):
    root_emb = graph.vec_of_ent(root_id)

    if depth == max_depth:
        return root_emb

    if root_id in emb_override:
        return emb_override[root_id]

    neighbors = graph.neighbors_of(root_id)[:max_neighbor]
    neighbor_features = np.zeros(0)

    for neighbor in neighbors:
        if neighbor.to_id in skip:
            continue

        feature = tf.concat([
            graph.vec_of_rel(neighbor.rel_id),
            recursive(
                graph,
                aggregator,
                neighbor.to_id,
                max_depth,
                max_neighbor,
                depth + 1
            )])
        neighbor_features = tf.stack(neighbor_features, feature)

    root_feature = aggregator(inputs=[root_emb, neighbor_features])

    return root_feature


def directional(graph, aggregator, path, width, max_neighbor):
    step_feature = None

    for ent_index in range(0, len(path), 2):

        if ent_index == 0:
            emb_override = {}
            skip = (path[ent_index + 2])
        elif ent_index == len(path) - 1:
            emb_override = {
                path[ent_index - 2]: step_feature
            }
            skip = (path[ent_index - 2])
        else:
            emb_override = {
                path[ent_index - 2]: step_feature
            }
            skip = (path[ent_index + 2])

        step_feature = recursive(
            graph=graph,
            aggregator=aggregator,
            root_id=path[ent_index],
            max_depth=width,
            max_neighbor=max_neighbor,
            skip=skip,
            emb_override=emb_override
        )

    return step_feature
