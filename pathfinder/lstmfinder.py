import tensorflow as tf


class LSTMFinder(object):
    def __init__(self, max_path_length, lstm_units, embedding_size):
        self.rnn_stack = {}
        self.selection_mlp = {}
        self.rnn_stack_depth = max_path_length
        self.embedding_size = embedding_size
        self.lstm_units = lstm_units

    def _get_path_history_cell(self):
        return tf.contrib.rnn.LSTMBlockCell(
            self.lstm_units, forget_bias=0.0)

    def build_history_stack(self):
        self.rnn_stack = tf.contrib.rnn.MultiRNNCell(
            [self._get_path_history_cell() for _ in range(self.rnn_stack_depth)])
