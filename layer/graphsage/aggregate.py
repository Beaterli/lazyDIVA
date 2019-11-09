import tensorflow as tf

from padding import zeros_tail_vec


def recursive(graph, aggregators, root_id, depth=0, skip=(), emb_override={}):
    root_emb = tf.expand_dims(graph.vec_of_ent(root_id), axis=0)
    root_emb_size = root_emb.shape.dims[1]

    if root_id in emb_override:
        return emb_override[root_id]

    if depth == len(aggregators):
        return root_emb

    aggregator = aggregators[depth]
    input_width = aggregator.get_input_width()

    neighbors = graph.neighbors_of(root_id)[:aggregators[depth].get_neighbor_size()]
    neighbor_features = []

    rel_emb_size = graph.vec_of_rel(0).size
    for neighbor in neighbors:
        if neighbor.to_id in skip:
            continue

        feature = tf.concat([
            tf.expand_dims(graph.vec_of_rel(neighbor.rel_id), axis=0),
            recursive(
                graph,
                aggregators,
                neighbor.to_id,
                depth + 1,
                skip + (root_id,),
                emb_override
            )], axis=1)

        feature = zeros_tail_vec(feature, input_width)

        neighbor_features.append(feature)

    if len(neighbor_features) == 0:
        return root_emb

    neighbor_features = tf.concat(neighbor_features, axis=0)
    padded_root_emb = tf.pad(root_emb, [[0, 0], [rel_emb_size, input_width - rel_emb_size - root_emb_size]])
    root_feature = aggregators[depth](inputs=[padded_root_emb, neighbor_features])

    return root_feature


def directional(graph, aggregators, path):
    step_feature = None

    for ent_index in range(0, len(path), 2):

        if ent_index == 0:
            emb_override = {}
            skip = (path[ent_index + 2],)
        elif ent_index == len(path) - 1:
            emb_override = {
                path[ent_index - 2]: step_feature
            }
            skip = ()
        else:
            emb_override = {
                path[ent_index - 2]: step_feature
            }
            skip = (path[ent_index + 2],)

        step_feature = recursive(
            graph=graph,
            aggregators=aggregators,
            root_id=path[ent_index],
            skip=skip,
            emb_override=emb_override
        )

    return step_feature[0]
