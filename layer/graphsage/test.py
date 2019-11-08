import tensorflow as tf

from graph.graph import Graph
from layer.graphsage.aggregate import directional
from layer.graphsage.layers import GraphConv

if __name__ == '__main__':
    test_graph_db = 'graph.db'
    graph = Graph(test_graph_db)
    graph.prohibit_relation('concept:athletehomestadium')
    path = [2592, 233, 16987, 275, 19365, 363, 3749]
    primary = GraphConv(
        input_feature_dim=2 * 100,
        output_feature_dim=1 * 100,
        neighbors=4,
        dtype=tf.float32
    )
    secondary = GraphConv(
        input_feature_dim=2 * 100,
        output_feature_dim=1 * 100,
        neighbors=4,
        dtype=tf.float32
    )
    print('---------------------before training---------------------')
    optimizer = tf.optimizers.Adam(1e-4)
    with tf.GradientTape() as tape:
        emb = directional(
            graph=graph,
            aggregators=[primary, secondary],
            path=path
        )
        print(primary.trainable_variables)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=[emb], labels=[tf.zeros(100)])
        gradient = tape.gradient(loss, primary.trainable_variables)
        optimizer.apply_gradients(zip(gradient, primary.trainable_variables))

    print('---------------------after training---------------------')
    print(primary.trainable_variables)
