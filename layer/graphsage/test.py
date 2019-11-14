import tensorflow as tf

from graph.graph import Graph
from layer.graphsage.aggregate import recursive
from layer.graphsage.layers import NeighborSampler, MaxPooling

if __name__ == '__main__':
    test_graph_db = 'nell995.db'
    graph = Graph(test_graph_db)
    graph.prohibit_relation('concept:athletehomestadium')
    path = [2592, 233, 16987, 275, 19365, 363, 3749]
    sampler = NeighborSampler(graph=graph)
    emb_size = 100
    # primary = GraphConv(
    #     input_feature_dim=3 * emb_size,
    #     output_feature_dim=3 * emb_size,
    #     dtype=tf.float32
    # )
    # secondary = GraphConv(
    #     input_feature_dim=2 * emb_size,
    #     output_feature_dim=2 * emb_size,
    #     dtype=tf.float32
    # )
    # print('---------------------before training---------------------')
    # optimizer = tf.optimizers.Adam(1e-4)
    # with tf.GradientTape() as tape:
    #     emb = recursive(
    #         graph=graph,
    #         sampler=sampler,
    #         aggregators=[primary, secondary],
    #         root_id=16987
    #     )
    #     print(primary.trainable_variables)
    #     loss = tf.nn.softmax_cross_entropy_with_logits(logits=[emb], labels=[tf.zeros(3 * emb_size)])
    #     gradient = tape.gradient(loss, primary.trainable_variables)
    #     optimizer.apply_gradients(zip(gradient, primary.trainable_variables))
    #
    # print('---------------------after training---------------------')
    # print(primary.trainable_variables)

    print('---------------------max pooling---------------------')
    max_pooling = MaxPooling(
        input_feature_dim=2 * emb_size,
        output_feature_dim=2 * emb_size,
        dtype=tf.float32
    )
    optimizer = tf.optimizers.Adam(1e-4)
    with tf.GradientTape() as tape:
        emb = recursive(
            graph=graph,
            sampler=sampler,
            aggregators=[max_pooling],
            root_id=16987
        )
        print('---------------------before training---------------------')
        print(max_pooling.trainable_variables)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=[emb], labels=[tf.zeros(2 * emb_size)])
        gradient = tape.gradient(loss, max_pooling.trainable_variables)
        optimizer.apply_gradients(zip(gradient, max_pooling.trainable_variables))
    print('---------------------after training---------------------')
    print(max_pooling.trainable_variables)
