'''
Utilities

2020-11-17 first created
'''

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from time import time, strftime, gmtime
import tensorflow as tf
tfk = tf.keras
tfkc = tfk.callbacks


class NBatchLogger(tfkc.Callback):
    '''A Logger that logs the average performance per `display` steps.

    See: https://gist.github.com/jaekookang/7e2ca4dc2b1ab10dbb80b9e65ca91179
    '''

    def __init__(self, n_display, max_epoch, save_dir=None, suffix=None, silent=False):
        self.epoch = 0
        self.display = n_display
        self.max_epoch = max_epoch
        self.logs = {}
        self.save_dir = save_dir
        self.silent = silent
        if self.save_dir is not None:
            assert os.path.exists(self.save_dir), Exception(
                f'Path:{self.save_dir} does not exist!')
            fname = 'train.log'
            if suffix is not None:
                fname = f'train_{suffix}.log'
            self.fid = open(os.path.join(save_dir, fname), 'w')
        self.t0 = time()

    def on_train_begin(self, logs={}):
        logs = logs or self.logs
        txt = f'=== Started at {self.get_time()} ==='
        self.write_log(txt)
        if not self.silent:
            print(txt)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if (self.epoch % self.display == 0) | (self.epoch == 1):
            txt = f' {self.get_time()} | Epoch: {self.epoch}/{self.max_epoch} | '
            if not self.silent:
                print(txt, end='')

            for i, key in enumerate(logs.keys()):
                if (i+1) == len(logs.keys()):
                    _txt = f'{key}={logs[key]:4f}'
                    if not self.silent:
                        print(_txt, end='\n')
                else:
                    _txt = f'{key}={logs[key]:4f} '
                    if not self.silent:
                        print(_txt, end='')
                txt = txt + _txt
            self.write_log(txt)
        self.logs = logs

    def on_train_end(self, logs={}):
        logs = logs or self.logs
        t1 = time()
        txt = f'=== Time elapsed: {(t1-self.t0)/60:.4f} min (loss:{logs["loss"]:4f}) ==='
        if not self.silent:
            print(txt)
        self.write_log(txt)

    def get_time(self):
        return strftime('%Y-%m-%d %Hh:%Mm:%Ss', gmtime())

    def write_log(self, txt):
        if self.save_dir is not None:
            self.fid.write(txt+'\n')
            self.fid.flush()


def visualize_moon(model, n_sample, direction='forward', sharexy=True, seed=0, xlim=[-4, 4], ylim=[-4, 4]):
    '''Visualize model prediction on moon-shaped dataset
    model: keras model
    n_sample: samples to generate
    directoin: 'forward' means data (x) to latent (z); default
               'inverse' means latent (z) to data (x)
    '''
    color_list = ['red', 'green', 'blue', 'black']

    if seed is not None:
        # if you want to control the randomness
        np.random.seed(seed)

    samples = []
    names = []
    if direction == 'forward':
        x2 = np.random.normal(0, 4, n_sample)
        x1 = np.random.normal(0.25 * x2**2, [1]*n_sample)
        moon = np.stack([x1, x2], axis=1).astype('float32')
        moon = StandardScaler().fit_transform(moon)
        samples.append(moon)
        names.append('data')
        x = moon
        for layer in model.layers:
            x = layer(x).numpy()
            samples.append(x)
            names.append(layer.name)

    elif direction == 'inverse':
        init_sample = np.random.randn(n_sample, 2).astype('float32')
        samples.append(init_sample)
        names.append('latent')
        x = init_sample
        for layer in reversed(model.layers):
            x = layer.inverse(x).numpy()
            samples.append(x)
            names.append(layer.name)

    fig, arr = plt.subplots(1, len(model.layers)+1, facecolor='white',
                            figsize=(3.5 * (len(model.layers)+1), 3.5),
                            sharex=sharexy, sharey=sharexy)

    # Divide 4 quadrants for tracing transformation
    X0 = samples[0]
    for i in range(len(samples)):
        X1 = samples[i]
        idx = np.logical_and(X1[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color=color_list[0])
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color=color_list[1])
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color=color_list[2])
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color=color_list[3])
        arr[i].set_title(names[i])
        if sharexy:
            arr[i].set_xlim(xlim)
            arr[i].set_ylim(ylim)
    return fig


def animate_moon_gif(model, n_sample, filename=None):
    # Make GIF file
    # - Do `brew install imagemagick` or `apt install imagemagick` first
    from matplotlib import animation
    from IPython.display import HTML
    n_layers = len(model.layers)
    interval = 100
    xlim = [-4, 4]
    ylim = [-4, 4]
    n_frm = 40

    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    z = np.random.randn(n_sample, 2).astype('float32')
    results = [z]
    for layer in reversed(model.layers):
        z = layer.inverse(z).numpy()
        results += [z]

    y = results[0]
    paths = ax.scatter(y[:, 0], y[:, 1], s=5, color='coral')

    def animate(i):
        n = n_frm
        l = i//n
        t = (float(i % n))/n
        try:
            y = (1-t)*results[l] + t*results[l+1]
        except:
            y = results[l]
        paths.set_offsets(y)
        return (paths,)

    anim = animation.FuncAnimation(
        fig, animate, frames=n_frm*(n_layers + 1), interval=interval, blit=False)
    if filename is not None:
        anim.save(filename, writer='imagemagick', fps=60)
        print(filename, 'saved')
    # HTML(anim.to_html5_video())
    return anim


def visualize_mnist_layers(model, dataset, direction='inverse', sharexy=True, cmap='gray'):
    '''Visualize layer activations
    model: NICE model
    dataset: tf dataset for MNIST
    direction: 'forward' means from data to latent 
               'inverse' means from latent to data
    **dataset** is required (tf dataset)
    '''
    samples = []
    names = []
    if direction == 'forward':
        it = iter(dataset)
        x = next(it)[0].numpy()[0]
        samples += [x]
        names += ['data']
        for layer in model.layers:
            x = layer(x)
            samples.append(x.numpy())
            names.append(layer.name)

    elif direction == 'inverse':
        n_samples = 784
        x = np.random.randn(n_samples) * 0.75
        x = x.astype('float32')
        samples += [x]
        names += ['latent']
        for layer in reversed(model.layers):
            x = layer(x)
            samples.append(x.numpy())
            names.append(layer.name)

    fig, arr = plt.subplots(1, len(samples), facecolor='white',
                            figsize=(4 * (len(samples)), 4),
                            sharex=sharexy, sharey=sharexy)

    cmap = matplotlib.cm.get_cmap(cmap)
    cmap = cmap.reversed()
    d = np.sqrt(x.shape[1]).astype('int')
    for i in range(len(samples)):
        X1 = samples[i].reshape(d, d)
        arr[i].imshow(X1, aspect='equal', cmap=cmap)
        arr[i].set_title(names[i])
    return fig


def animate_mnist_gif(model, filename=None):
    # Make GIF file
    # - Do `brew install imagemagick` or `apt install imagemagick` first
    from matplotlib import animation
    from IPython.display import HTML
    interval = 50   # ms
    n_layers = len(model.layers)
    n_dim = 784
    h, w = 28, 28
    xlim = [-4, 4]
    ylim = [-4, 4]
    n_frm = 30

    fig, ax = plt.subplots()

    z = np.random.randn(1, n_dim).astype('float32') * 0.75
    results = [z]
    for layer in reversed(model.layers):
        z = layer.inverse(z).numpy()
        results += [z]
    results = [d.reshape(w, h) for d in results]

    y = results[0]
    imgs = ax.imshow(y, aspect='equal', cmap='gray')

    def animate(i):
        n = n_frm
        l = i//n
        t = (float(i % n))/n
        try:
            y = (1-t)*results[l] + t*results[l+1]
        except:
            y = results[l]
        imgs.set_data(y)
        return (imgs,)

    anim = animation.FuncAnimation(
        fig, animate, frames=n_frm*(n_layers + 1), interval=interval, blit=False)
    if filename is not None:
        anim.save(filename, writer='imagemagick', fps=60)
        print(filename, 'saved')
    # HTML(anim.to_html5_video())
    return anim
