'''Module of various Tensorflow models and associated functions

'''
import tensorflow as tf


LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on',
          'off', 'stop', 'go', 'silence', 'unknown']
SAMPLE_INPUT_LENGTH = 16000
SAMPLE_RATE = 16000


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


def mfcc_spectrogram_cnn():
    x = tf.placeholder(tf.float32, shape=[None, SAMPLE_INPUT_LENGTH])
    s = tf.placeholder(tf.int32, shape=[None, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, len(LABELS)])

    # MFCC related code stolen from
    # https://tensorflow.org/api_guides/python/contrib.signal#Computing_spectrograms
    stfts = tf.contrib.signal.stft(x, frame_length=256, frame_step=128,
                                   fft_length=1024)
    magnitude_spectrograms = tf.abs(stfts)

    num_spectrograms_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 64
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrograms_bins, SAMPLE_RATE,
        lower_edge_hertz, upper_edge_hertz
    )

    mel_spectrograms = tf.tensordot(magnitude_spectrograms,
                                    linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
    num_mfccs = 13
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfccs]

    image_size = [mfccs.shape[-2].value, mfccs.shape[-1].value]
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    mfcc_images = tf.reshape(mfccs, [-1] + image_size + [1])
    h_conv1 = tf.nn.relu(conv2d(mfcc_images, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    flatten_size = (h_pool2.shape[-3].value *
                    h_pool2.shape[-2].value *
                    h_pool2.shape[-1].value)
    W_fc1 = weight_variable([flatten_size, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, flatten_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, len(LABELS)])
    b_fc2 = bias_variable([len(LABELS)])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    return {
        'train_step': train_step,
        'x': x,
        'y': y_,
        's': s,
        'keep_prob': keep_prob,
        'y_predict': prediction,
    }
