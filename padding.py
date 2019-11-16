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


def zeros_bottom(mat, height):
    mat_height = mat.shape.dims[0]
    if mat_height < height:
        return tf.pad(mat, [[0, height - mat_height], [0, 0]])
    return mat


if __name__ == '__main__':
    original = tf.constant([[1, 2, 3]])
    print(zeros_front_vec(original, 5))
    print(zeros_tail_vec(original, 5))
    mat = tf.constant([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(zeros_bottom(mat, 5))
