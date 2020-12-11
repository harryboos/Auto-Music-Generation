import tensorflow as tf
from tensorflow.python import keras
import numpy as np
import math

class RelativeGlobalAttention(keras.layers.Layer):
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__(**kwargs)
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = keras.layers.Dense(int(self.d))
        self.Wk = keras.layers.Dense(int(self.d))
        self.Wv = keras.layers.Dense(int(self.d))
        self.fc = keras.layers.Dense(d)
        self.additional = add_emb
        if self.additional:
            self.Radd = None

    def build(self, input_shape):
        self.shape_q = input_shape[0][1]
        self.shape_k = input_shape[1][1]
        self.E = self.add_weight('emb', shape=[self.max_seq, int(self.dh)])

    def call(self, inputs, mask=None, **kwargs):
       
        q = inputs[0]
        q = self.Wq(q)
        q = tf.reshape(q, (q.shape[0], q.shape[1], self.h, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = tf.reshape(k, (k.shape[0], k.shape[1], self.h, -1))
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs[2]
        v = self.Wv(v)
        v = tf.reshape(v, (v.shape[0], v.shape[1], self.h, -1))
        v = tf.transpose(v, (0, 2, 1, 3))

        self.len_k = k.shape[2]
        self.len_q = q.shape[2]

        E = self._get_left_embedding(self.len_q, self.len_k)
        QE = tf.einsum('bhld,md->bhlm', q, E)
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = tf.transpose(k,[0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (tf.cast(mask, tf.float32) * -1e9)

        attention_weights = tf.nn.softmax(logits, -1)

        attention = tf.matmul(attention_weights, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (out.shape[0], -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    @staticmethod
    def _qe_masking(qe):
        mask = tf.sequence_mask(
            tf.range(qe.shape[-1] -1, qe.shape[-1] - qe.shape[-2] -1, -1), qe.shape[-1])

        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)

        return mask * qe

    def _skewing(self, tensor: tf.Tensor):
        padded = tf.pad(tensor, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, shape=[-1, padded.shape[1], padded.shape[-1], padded.shape[-2]])
        Srel = reshaped[:, :, 1:, :]

        if self.len_k > self.len_q:
            Srel = tf.pad(Srel, [[0,0], [0,0], [0,0], [0, self.len_k-self.len_q]])
        elif self.len_k < self.len_q:
            Srel = Srel[:,:,:,:self.len_k]

        return Srel
