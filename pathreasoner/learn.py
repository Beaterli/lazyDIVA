import tensorflow as tf


def learn_from_path(reasoner, path, label):
    with tf.GradientTape() as tape:
        relation = reasoner.relation_of_path(path)
        cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=[relation], labels=[label])
        # 分类结果熵向量求和
        classify_loss = tf.reduce_mean(cross_ent)

    return classify_loss, tape.gradient(classify_loss, reasoner.trainable_variables)


def learn_from_paths(reasoner, paths, label):
    with tf.GradientTape() as tape:
        relations = []
        labels = []
        for path in paths:
            relations.append(reasoner.relation_of_path(path))
            labels.append(label)
        cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=relations, labels=labels)
        # 分类结果熵向量求和
        classify_loss = tf.reduce_mean(cross_ent, axis=[0])

    return classify_loss, tape.gradient(classify_loss, reasoner.trainable_variables)
