import tensorflow as tf


def recursive(graph, aggregators, root_id, depth=0, skip=(), emb_override={}):
    root_emb = graph.vec_of_ent(root_id)

    if depth == len(aggregators):
        return tf.expand_dims(root_emb, axis=0)

    if root_id in emb_override:
        return emb_override[root_id]

    aggregator = aggregators[depth]
    input_width = aggregator.get_input_width()

    neighbors = graph.neighbors_of(root_id)[:aggregators[depth].get_neighbor_size()]
    neighbor_features = []

    for neighbor in neighbors:
        if neighbor.to_id in skip:
            continue

        feature = tf.concat([
            graph.vec_of_rel(neighbor.rel_id),
            recursive(
                graph,
                aggregators,
                neighbor.to_id,
                depth + 1,
                skip + (root_id,),
                emb_override
            )[0]], axis=0)

        neighbor_features.append(feature)

    neighbor_features = tf.stack(neighbor_features)
    padded_root_emb = tf.concat([tf.zeros(input_width - root_emb.size, tf.float32), root_emb], axis=0)
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
            skip = (path[ent_index - 2],)
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
