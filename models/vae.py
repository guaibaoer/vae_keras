"""

References:
    Paper:
        Kingma D P, Welling M. Auto-Encoding Variational Bayes[J]. 2013.
        https://arxiv.org/abs/1312.6114
    论文笔记：
        变分自编码器（一）：原来是这么一回事 - 科学空间|Scientific Spaces https://kexue.fm/archives/5253
    code:
        - https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
        - https://github.com/bojone/vae/blob/master/vae_keras.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from utils import plot_results
from data_helper import load_mnist

import keras.backend as K
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Lambda

OUT_PATH_PREFIX = 'out/vea'
os.makedirs(OUT_PATH_PREFIX, exist_ok=True)


class config:
    """"""
    input_shape = None
    intermediate_dim = 512
    latent_dim = 2  # 隐变量取 2 维是为了方便之后作图

    mse = False

    # train
    batch_size = 128
    epochs = 50


def build_model():
    """"""
    # VAE model = encoder + decoder

    # encoder 部分
    inputs = Input(shape=config.input_shape, name='encoder_input')
    x = Dense(config.intermediate_dim, activation='relu')(inputs)

    # 算p(Z|X)的均值和方差
    z_mean = Dense(config.latent_dim, name='z_mean')(x)
    z_log_var = Dense(config.latent_dim, name='z_log_var')(x)

    def sampling(args):
        """ 重参数(reparameterization)技巧"""
        z_mean, z_log_var = args
        batch_size, latent_dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size, latent_dim))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # 重参数层，相当于
    z = Lambda(sampling, output_shape=(config.latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # 打印 encoder model
    plot_model(encoder, to_file=os.path.join(OUT_PATH_PREFIX, 'vae_mlp_encoder.png'), show_shapes=True)

    # decoder 部分
    latent_inputs = Input(shape=(config.latent_dim,), name='z_sampling')
    x = Dense(config.intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # 打印 dncoder model
    plot_model(decoder, to_file=os.path.join(OUT_PATH_PREFIX, 'vae_mlp_decoder.png'), show_shapes=True)

    # VAE 部分
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    # vea_loss = mse_loss or xent_loss + kl_loss
    if config.mse:
        from keras.losses import mse
        reconstruction_loss = mse(inputs, outputs)
    else:
        from keras.losses import binary_crossentropy
        reconstruction_loss = binary_crossentropy(inputs, outputs)
    xent_loss = original_dim * reconstruction_loss
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    plot_model(vae, to_file=os.path.join(OUT_PATH_PREFIX, 'vae_mlp.png'), show_shapes=True)

    return vae


if __name__ == '__main__':
    """"""
    x_train, x_test, original_dim = load_mnist()

    config.input_shape = (original_dim,)

    vae = build_model()

    vae.fit(x_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=(x_test, None),
            verbose=2)

    vae.save_weights(os.path.join(OUT_PATH_PREFIX, 'vae_mlp_mnist.h5'))
