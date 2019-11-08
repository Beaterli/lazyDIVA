import tensorflow as tf


class GraphConv(tf.keras.layers.Layer):
    """
    from original GraphSAGE implementation, adapt to tensorflow 2.0
    """

    def __init__(self,
                 root_feature_dim,
                 out_feature_dim,
                 neighbor_feature_dim=None,
                 dropout=0.,
                 bias=False,
                 act=tf.nn.relu,
                 name='GraphConv',
                 dtype=None,
                 dynamic=False,
                 **kwargs):

        super(GraphConv, self).__init__(
            trainable=True,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs)

        self.dropout = dropout
        self.use_bias = bias
        self.act = act

        if neighbor_feature_dim is None:
            neigh_input_dim = root_feature_dim

        self.weights = self.add_weight(
            name='neighbor_weights',
            dtype=dtype,
            shape=(neighbor_feature_dim, out_feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_dim,),
                initializer='zeros',
                trainable=True
            )

        self.root_feature_dim = root_feature_dim
        self.out_feature_dim = out_feature_dim

    def call(self, root_and_neighbors):
        root_feature, neighbor_features = root_and_neighbors

        neighbor_features = tf.nn.dropout(neighbor_features, 1 - self.dropout)
        root_feature = tf.nn.dropout(root_feature, 1 - self.dropout)

        means = tf.reduce_mean(tf.concat([
            neighbor_features,
            tf.expand_dims(root_feature, axis=1)
        ], axis=1), axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.weights)

        # bias
        if self.use_bias:
            output += self.bias

        return self.act(output)
