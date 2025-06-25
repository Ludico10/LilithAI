import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers


class DuelingCNN(keras.Model):
    def __init__(self, hidden_dims, output_dim, name="SnakeNN", **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.conv1 = layers.Conv2D(3, 3, activation='relu')
        self.pool1 = layers.AveragePooling2D(2)
        self.conv2 = layers.Conv2D(3, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.concat = layers.Concatenate()

        self.hidden_layers = []
        for dim in self.hidden_dims:
            self.hidden_layers.append(layers.Dense(dim, activation='relu'))

        self.output_v = layers.Dense(1)
        self.output_a = layers.Dense(output_dim)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.concat([x, self.flatten(inputs)])

        for dense_layer in self.hidden_layers:
            x = dense_layer(x)

        v = self.output_v(x)
        a = self.output_a(x)
        q = v + a - tf.reduce_mean(a)
        return q