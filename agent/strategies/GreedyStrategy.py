import tensorflow as tf


class GreedyStrategy:
    def select_action(self, model, state):
        return tf.argmax(tf.squeeze(model(state), axis=0)).numpy()
