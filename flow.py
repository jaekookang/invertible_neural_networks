'''
Flow based models

- NICE
- RealNVP

2020-11-18 first created
'''

import tensorflow as tf
from utils import *

tfk = tf.keras
tfkl = tfk.layers
tfkc = tfk.callbacks
K = tfk.backend


def fully_connected(n_dim, n_layer=3, n_hid=512, activaiton='relu'):
    '''Fully connected neural networks'''
    nn = tfk.Sequential(name='neural_net')
    for _ in range(n_layer - 1):
        nn.add(tfkl.Dense(n_hid, activation=activaiton))
    nn.add(tfkl.Dense(n_dim//2, activation='linear'))
    return nn


class NN(tfkl.Layer):
    def __init__(self, n_dim, n_layer=3, n_hid=512, activation='relu', name='fc_layer'):
        super(NN, self).__init__(name=name)
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.n_hid = n_hid
        self.layer_list = []
        for _ in range(n_layer):
            self.layer_list.append(tfkl.Dense(n_hid, activation=activation))
        self.log_s_layer = tfkl.Dense(
            n_dim//2, activation='tanh', name='log_s_layer')
        self.t_layer = tfkl.Dense(
            n_dim//2, activation='linear', name='t_layer')

    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        log_s = self.log_s_layer(x)
        t = self.t_layer(x)
        return log_s, t


class AdditiveCouplingLayer(tfkl.Layer):
    '''Implementation of Additive Coupling layers in Dinh et al (2015)

    # forward
    y1 = x1
    y2 = x2 + m(x1)
    # inverse
    x1 = y1
    x2 = y2 - m(y1)
    '''

    def __init__(self, inp_dim, n_hid_layer, n_hid_dim, name, shuffle_type='reverse'):
        super(AdditiveCouplingLayer, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.m = fully_connected(inp_dim, n_hid_layer, n_hid_dim)
        self.idx = tf.Variable(list(range(inp_dim)),
                               shape=(inp_dim,),
                               trainable=False,
                               name='index',
                               dtype='int64')
        if self.shuffle_type == 'random':
            self.idx.assign(tf.random.shuffle(self.idx))
        elif self.shuffle_type == 'reverse':
            self.idx.assign(tf.reverse(self.idx, axis=[0]))
        else:
            raise NotImplementedError

    def call(self, x):
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        mx1 = self.m(x1)
        y1 = x1
        y2 = x2 + mx1
        y = tf.concat([y1, y2], axis=-1)
        return y

    def inverse(self, y):
        y1, y2 = self.split(y)
        my1 = self.m(y1)
        x1 = y1
        x2 = y2 - my1
        x = tf.concat([x1, x2], axis=-1)
        x = self.shuffle(x, isInverse=True)
        return x

    def split(self, x):
        dim = self.inp_dim
        x = tf.reshape(x, [-1, dim])
        return x[:, :dim//2], x[:, dim//2:]

    def shuffle(self, x, isInverse=False):
        if not isInverse:
            # Forward
            idx = self.idx
        else:
            # Inverse
            idx = tf.map_fn(tf.math.invert_permutation,
                            tf.expand_dims(self.idx, 0))
            idx = tf.squeeze(idx)
        x = tf.transpose(x)
        x = tf.gather(x, idx)
        x = tf.transpose(x)
        return x


class NVPCouplingLayer(tfkl.Layer):
    def __init__(self, inp_dim, n_hid_layer, n_hid_dim, name, shuffle_type='reverse'):
        super(NVPCouplingLayer, self).__init__(name=name)
        '''Implementation of Coupling layers in Dinh et al (2017)

        # Forward
        y1 = x1
        y2 = x2 * exp(s(x1)) + t(x2)
        # Inverse
        x1 = y1
        x2 = (y2 - t(y1)) / exp(s(y1))
        '''
        self.inp_dim = inp_dim
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.nn = NN(inp_dim, n_hid_layer, n_hid_dim)
        self.idx = tf.Variable(list(range(inp_dim)),
                               shape=(inp_dim,),
                               trainable=False,
                               name='index',
                               dtype='int64')
        if self.shuffle_type == 'random':
            self.idx.assign(tf.random.shuffle(self.idx))
        elif self.shuffle_type == 'reverse':
            self.idx.assign(tf.reverse(self.idx, axis=[0]))
        else:
            raise NotImplementedError

    def call(self, x):
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        log_s, t = self.nn(x1)
        y1 = x1
        y2 = x2 * tf.math.exp(log_s) + t
        y = tf.concat([y1, y2], axis=-1)
        # Add loss
        self.log_det_J = log_s
        self.add_loss(- tf.math.reduce_sum(self.log_det_J))
        return y

    def inverse(self, y):
        y1, y2 = self.split(y)
        log_s, t = self.nn(y1)
        x1 = y1
        x2 = (y2 - t)/tf.math.exp(log_s)
        x = tf.concat([x1, x2], axis=-1)
        x = self.shuffle(x, isInverse=True)
        return x

    def split(self, x):
        dim = self.inp_dim
        x = tf.reshape(x, [-1, dim])
        return x[:, :dim//2], x[:, dim//2:]

    def shuffle(self, x, isInverse=False):
        if not isInverse:
            # Forward
            idx = self.idx
        else:
            # Inverse
            idx = tf.map_fn(tf.math.invert_permutation,
                            tf.expand_dims(self.idx, 0))
            idx = tf.squeeze(idx)
        x = tf.transpose(x)
        x = tf.gather(x, idx)
        x = tf.transpose(x)
        return x


class ScalingLayer(tfkl.Layer):
    def __init__(self, inp_dim):
        super(ScalingLayer, self).__init__(name='ScalingLayer')
        self.inp_dim = inp_dim
        self.scaling = tf.Variable(shape=(inp_dim,),
                                   trainable=True,
                                   initial_value=tfk.initializers.glorot_uniform()((inp_dim,)))

    def call(self, x):
        self.add_loss(-tf.math.reduce_sum(self.scaling))
        return tf.math.exp(self.scaling) * x

    def inverse(self, y):
        return tf.math.exp(-self.scaling) * y


class RealNVP(tfk.Model):
    def __init__(self, inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, name, shuffle_type='reverse'):
        super(RealNVP, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = NVPCouplingLayer(
                inp_dim, n_hid_layer, n_hid_dim, name=f'Layer{i}', shuffle_type=shuffle_type)
            self.AffineLayers.append(layer)

    def call(self, x):
        '''Forward: data (x) --> latent (z); inference'''
        z = x
        for i in range(self.n_couple_layer):
            z = self.AffineLayers[i](z)
        return z

    def inverse(self, z):
        '''Inverse: latent (z) --> data (y); sampling'''
        x = z
        for i in reversed(range(self.n_couple_layer)):
            x = self.AffineLayers[i].inverse(x)
        return x


class NICE(tfk.Model):
    def __init__(self, inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, name, shuffle_type='reverse'):
        super(NICE, self).__init__(name=name)
        self.inp_dim = inp_dim
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = AdditiveCouplingLayer(
                inp_dim, n_hid_layer, n_hid_dim, name=f'Layer{i}', shuffle_type=shuffle_type,)
            self.AffineLayers.append(layer)
        self.scale = ScalingLayer(inp_dim)
        self.AffineLayers.append(self.scale)

    def call(self, x):
        '''Forward: data (x) --> latent (z); inference'''
        z = x
        for i in range(self.n_couple_layer):
            z = self.AffineLayers[i](z)
        z = self.scale(z)
        return z

    def inverse(self, z):
        '''Inverse: latent (z) --> data (y); sampling'''
        x = self.scale.inverse(z)
        for i in reversed(range(self.n_couple_layer)):
            x = self.AffineLayers[i].inverse(x)
        return x


if __name__ == "__main__":
    inp_dim = 2
    n_couple_layer = 3
    n_hid_layer = 3
    n_hid_dim = 512

    model1 = NICE(inp_dim, n_couple_layer, n_hid_layer, n_hid_dim)
    x = tfkl.Input(shape=(inp_dim,))
    model1(x)
    model1.summary()

    model2 = RealNVP(inp_dim, n_couple_layer, n_hid_layer, n_hid_dim)
    x = tfkl.Input(shape=(inp_dim,))
    model2(x)
    model2.summary()
