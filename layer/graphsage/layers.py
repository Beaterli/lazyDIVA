import random

import tensorflow as tf

from padding import zeros_bottom


class NeighborSampler(tf.keras.layers.Layer):
    def __init__(self,
                 graph,
                 random_sample=True,
                 name='RandomNeighborSampler',
                 **kwargs):
        super(NeighborSampler, self).__init__(
            trainable=False,
            name=name,
            **kwargs)
        self.graph = graph
        self.random_sample = random_sample

    def call(self, inputs):
        ent_id, skips, count = inputs
        neighbors = self.graph.neighbors_of(ent_id).copy()
        if self.random_sample:
            random.shuffle(neighbors)
        if count is None:
            return neighbors
        else:
            return neighbors[:count]


class GraphConv(tf.keras.layers.Layer):
    """
    from original GraphSAGE implementation, adapt to tensorflow 2.0.

    输入邻居矩阵形式为(邻居数)*(特征维度)
    计算流程为先纵向取所有邻居节点在某个特征维度的平均值（纵向平均），再与权重矩阵相乘得到聚合特征值。
    在测试中学习速度快于求所有邻居节点自身所有维度的平均值（横向平均）
    """

    def __init__(self,
                 input_feature_dim,
                 output_feature_dim,
                 vertical_mean=True,
                 neighbors=None,
                 dropout=None,
                 bias=False,
                 act=tf.nn.relu,
                 name='GraphConv',
                 dtype=None,
                 dynamic=True,
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
        self.vertical_mean = vertical_mean

        if vertical_mean:
            w_dims = (input_feature_dim, output_feature_dim)
        else:
            w_dims = (self.neighbors + 1, output_feature_dim)
        self.w = self.add_weight(
            name='neighbor_weights',
            shape=w_dims,
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

    def call(self, inputs):
        root_feature, neighbor_features = inputs
        neighbors = neighbor_features.get_shape().dims[0]

        if not self.vertical_mean:
            if neighbors == 0:
                neighbor_features = tf.zeros([self.neighbors, self.input_feature_dim], tf.float32)
            elif neighbors < self.neighbors:
                neighbor_features = zeros_bottom(neighbor_features, self.neighbors)

        if self.dropout is not None:
            neighbor_features = tf.nn.dropout(neighbor_features, 1 - self.dropout)
            root_feature = tf.nn.dropout(root_feature, 1 - self.dropout)

        concated_features = tf.concat(
            [root_feature, neighbor_features],
            axis=0
        )

        if self.vertical_mean:
            mean_axis = 0

        else:
            mean_axis = 1
        mean_features = tf.expand_dims(
            tf.math.reduce_mean(concated_features, axis=mean_axis),
            axis=0)

        # [nodes] x [out_dim]
        output = tf.matmul(mean_features, self.w)

        # bias
        if self.use_bias:
            output += self.b

        return self.act(output)


if __name__ == '__main__':
    neighbor_tensor = tf.constant([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 11.0, 12.0, 13.0],
        [18.0, 19.0, 20.0, 21.0]
    ])
    neighbor_tensor = zeros_bottom(neighbor_tensor, 4)

    root_tensor = tf.constant([[-1.0, -2.0, -3.0, -4.0]])
    print('root: {}'.format(root_tensor))
    concated_tensor = tf.concat(
        [root_tensor, neighbor_tensor],
        axis=0
    )
    print('concat: {}'.format(concated_tensor))
    shape = concated_tensor.shape
    print('{} neighbors'.format(shape.dims[0]))
    means = tf.expand_dims(
        tf.math.reduce_mean(
            concated_tensor,
            axis=0
        ),
        axis=0)
    print('means: {}'.format(means))
    w = tf.constant([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ])
    print(tf.matmul(means, w))
