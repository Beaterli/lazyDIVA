import tensorflow as tf


def zeros_front_vec(vec, width):
    vec_width = vec.shape.dims[1]
    if vec_width < width:
        return tf.pad(vec, [[0, 0], [width - vec_width, 0]])
    return vec


def zeros_tail_vec(vec, width):
    vec_width = vec.shape.dims[1]
    if vec_width < width:
        return tf.pad(vec, [[0, 0], [0, width - vec_width]])
    return vec


if __name__ == '__main__':
    original = tf.constant([[1, 2, 3]])
    print(zeros_front_vec(original, 5))
    print(zeros_tail_vec(original, 5))
