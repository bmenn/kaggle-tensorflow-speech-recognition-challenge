import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from .resnet import resnet_model as resnet_model
from .constants import *


# Helper functions stolen from
# https://tensorflow.org/api_guides/python/contrib.signal#Computing_spectrograms
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def input_tensors():
    with tf.variable_scope('training'):
        x = tf.placeholder(tf.float32, shape=[None, SAMPLE_INPUT_LENGTH],
                           name='wav_input')
        s = tf.placeholder(tf.int32, shape=[None, 1],
                           name='sample_rate')
        y_ = tf.placeholder(tf.float32, shape=[None, len(LABELS)],
                            name='label')
        is_training = tf.placeholder(tf.bool, name='is_training')
    return {
        'wav_input': x,
        'sample_rate': s,
        'label': y_,
        'is_training': is_training,
    }


def log_mel_spectrogram(x, s, frame_length=480, frame_step=160,
                        fft_length=None, lower_hertz=20.0,
                        upper_hertz=4000.0, num_mel_bins=40):
    # MFCC related code stolen from
    # https://tensorflow.org/api_guides/python/contrib.signal#Computing_spectrograms
    stfts = tf.contrib.signal.stft(x, frame_length=frame_length,
                                   frame_step=frame_step,
                                   fft_length=fft_length)
    magnitude_spectrograms = tf.abs(stfts)

    num_spectrograms_bins = magnitude_spectrograms.shape[-1].value

    # TODO Ideally SAMPLE_RATE would be a tensor, `s`, instead of a constant
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrograms_bins, SAMPLE_RATE,
        lower_hertz, upper_hertz
    )

    mel_spectrograms = tf.tensordot(magnitude_spectrograms,
                                    linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

    log_offset = 1e-6
    return tf.log(mel_spectrograms + log_offset)

