'''Invertible Neural Networks

2020-11-26 first created
'''

import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
K = tfk.backend


class NN(tfkl.Layer):
    '''Fully-connected neural-net layer
    Retrieved from https://github.com/MokkeMeguru/glow-realnvp-tutorial
    '''
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


class NVPCouplingLayer(tfkl.Layer):
    def __init__(self, inp_dim, n_hid_layer, n_hid_dim, name):
        super(NVPCouplingLayer, self).__init__(name=name)
        '''Implementation of two complimentary Affine Coupling Layers 
        in Ardizzone et al. (2019)

        This layer includes both
        - coupling layer and
        - permutation

        # Forward
        y1 = x1 * exp(s2(x2)) + t2(x2)
        y2 = x2 * exp(s1(y1)) + t1(y1)
        # Inverse
        x1 = (y2 - t1(y1)) * exp(-s1(y1))
        x2 = (y1 - t2(x2)) * exp(-s2(x2))
        '''
        self.inp_dim = inp_dim
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.nn1 = NN(inp_dim, n_hid_layer, n_hid_dim)
        self.nn2 = NN(inp_dim, n_hid_layer, n_hid_dim)
        self.permute = tfkl.Lambda(lambda x: tf.gather(
            x, list(reversed(range(x.shape[-1]))), axis=-1))

    def call(self, x):
        x = self.permute(x)
        x1, x2 = tf.split(x, 2, axis=-1)
        log_s2, t2 = self.nn2(x2)
        y1 = x1 * tf.math.exp(log_s2) + t2
        log_s1, t1 = self.nn2(y1)
        y2 = x2 * tf.math.exp(log_s1) + t1
        y = tf.concat([y1, y2], axis=-1)
        # Add loss
        self.log_det_J = log_s1 * log_s2
        self.add_loss(- tf.math.reduce_sum(self.log_det_J))
        return y

    def inverse(self, y):
        y1, y2 = tf.split(y, 2, axis=-1)
        log_s1, t1 = self.nn1(y1)
        x2 = (y2 - t1) * tf.math.exp(-log_s1)
        log_s2, t2 = self.nn2(x2)
        x1 = (y1 - t2) * tf.math.exp(-log_s2)
        x = tf.concat([x1, x2], axis=-1)
        x = self.permute(x)
        return x


class SigmoidLayer(tfkl.Layer):
    def __init__(self, inp_dim, z_dim):
        super(SigmoidLayer, self).__init__(name='SigmoidLayer')
        self.inp_dim = inp_dim
        self.z_dim = z_dim

    def call(self, x):
        # x is assumed to be [z, pad, y] where z is at the end
        z, x = x[:, :self.z_dim], x[:, self.z_dim:]  # split
        x_sigmoid = tfk.activations.sigmoid(x)
        x = tf.concat([z, x_sigmoid], axis=-1) # combine
        return x

    def inverse(self, x):
        # x is assumed to be [z, pad, y] where z is at the end
        z, x_sigmoid = x[:, :self.z_dim], x[:, self.z_dim:] # split
        x_sigmoid = tf.clip_by_value(x_sigmoid, K.epsilon(), 1. - K.epsilon())
        x = tf.math.log(tf.math.divide(x_sigmoid, (1. - x_sigmoid)))
        x = tf.concat([z, x], axis=-1) # combine
        return x        


class InvertibleClassifierNet(tfk.Model):
    def __init__(self, inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, z_dim, **kwargs):
        super(InvertibleClassifierNet, self).__init__(**kwargs)
        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = NVPCouplingLayer(
                inp_dim, n_hid_layer, n_hid_dim, name=f'Layer{i}')
            self.AffineLayers.append(layer)
        self.SigmoidLayer = SigmoidLayer(inp_dim, z_dim)

    def call(self, x):
        '''Forward: data (x) --> latent (z); inference'''
        z = x
        for i in range(self.n_couple_layer):
            z = self.AffineLayers[i](z)
        z = self.SigmoidLayer(z)
        return z

    def inverse(self, z):
        '''Inverse: latent (z) --> data (y); sampling'''
        x = z
        for i in reversed(range(self.n_couple_layer)):
            x = self.AffineLayers[i].inverse(x)
        x = self.SigmoidLayer.inverse(x)
        return x

class InvertibleRegressorNet(tfk.Model):
    def __init__(self, inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, **kwargs):
        super(InvertibleRegressorNet, self).__init__(**kwargs)
        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.AffineLayers = []
        for i in range(n_couple_layer):
            layer = NVPCouplingLayer(
                inp_dim, n_hid_layer, n_hid_dim, name=f'Layer{i}')
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


if __name__ == '__main__':
    inp_dim = 2
    n_hid_layer = 3
    n_hid_dim = 512
    n_couple_layer = 3

    NVP = NVPCouplingLayer(inp_dim, n_hid_layer, n_hid_dim, name='nvp')
    x = tf.random.normal((5,4))
    y = NVP(x)

    INN = InvertibleNet(inp_dim, n_couple_layer, n_hid_layer, n_hid_dim, name='INN')
    y = INN(x)
    INN.summary()
    pass
