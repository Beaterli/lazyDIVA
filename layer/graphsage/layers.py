import tensorflow as tf


class GraphConv(tf.keras.layers.Layer):
    """
    from original GraphSAGE implementation, adapt to tensorflow 2.0
    """

    def __init__(self,
                 input_feature_dim,
                 output_feature_dim,
                 neighbors,
                 dropout=None,
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
        self.neighbors = neighbors

        self.w = self.add_weight(
            name='neighbor_weights',
            shape=(self.neighbors + 1, output_feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        if self.use_bias:
            self.b = self.add_weight(
                name='output_bias',
                shape=(output_feature_dim,),
                initializer='zeros',
                trainable=True
            )

        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim

    def get_neighbor_size(self):
        return self.neighbors

    def get_input_width(self):
        return self.input_feature_dim

    def get_output_width(self):
        return self.output_feature_dim

    '''
        root feature shape is 1*emb_size
        neighbor features shape is n*emb_size
    '''
    def call(self, root_and_neighbors):
        root_feature, neighbor_features = root_and_neighbors
        neighbors = neighbor_features.get_shape().dims[0]
        if neighbors == 0:
            neighbor_features = tf.zeros([self.neighbors, self.input_feature_dim], tf.float32)
        elif neighbors < self.neighbors:
            neighbor_features = tf.concat(
                [neighbor_features,
                 tf.zeros(
                     [self.neighbors - neighbors,
                      self.input_feature_dim],
                     tf.float32)],
                axis=0
            )

        if self.dropout is not None:
            neighbor_features = tf.nn.dropout(neighbor_features, 1 - self.dropout)
            root_feature = tf.nn.dropout(root_feature, 1 - self.dropout)

        root_feature = tf.expand_dims(root_feature, axis=0)
        concated_features = tf.concat(
            [root_feature, neighbor_features],
            axis=0
        )
        mean_features = tf.expand_dims(
            tf.math.reduce_mean(concated_features, axis=1),
            axis=0)

        # [nodes] x [out_dim]
        output = tf.matmul(mean_features, self.w)

        # bias
        if self.use_bias:
            output += self.b

        return self.act(output)


if __name__ == '__main__':
    neighbor_tensor = [
        [1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0],
        [18.0, 19.0, 20.0]
    ]
    neighbor_tensor = tf.concat(
        [neighbor_tensor, tf.zeros([1, 3], tf.float32)], axis=0
    )
    root_tensor = tf.expand_dims([-1.0, -2.0, -3.0], axis=0)
    print(root_tensor)
    concated_tensor = tf.concat(
        [root_tensor, neighbor_tensor],
        axis=0
    )
    print(concated_tensor)
    shape = concated_tensor.shape
    print('{} neighbors'.format(shape.dims[0]))
    means = tf.expand_dims(
        tf.math.reduce_mean(
            concated_tensor,
            axis=1
        ),
        axis=0)
    print(means)
    w = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0]
    ]
    print(tf.matmul(means, w))
