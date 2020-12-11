import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import math

class DynamicPositionEmbedding(keras.layers.Layer):
    def __init__(self, embedding_dim, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = np.array([[
            [
                math.sin(
                    pos * math.exp(-math.log(10000) * i/embedding_dim) *
                    math.exp(math.log(10000)/embedding_dim * (i % 2)) + 0.5 * math.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding[:,:inputs.shape[1],:])