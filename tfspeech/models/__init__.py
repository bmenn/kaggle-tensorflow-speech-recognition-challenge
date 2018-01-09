'''Module of various Tensorflow models and associated functions

'''
import tensorflow as tf

from .resnet import resnet_model as resnet_model


LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on',
          'off', 'stop', 'go', 'silence', 'unknown']
SAMPLE_INPUT_LENGTH = 16000
SAMPLE_RATE = 16000

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4


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
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    with tf.variable_scope('training'):
        x = tf.placeholder(tf.float32, shape=[None, SAMPLE_INPUT_LENGTH],
                           name='wav_input')
        s = tf.placeholder(tf.int32, shape=[None, 1],
                           name='sample_rate')
        y_ = tf.placeholder(tf.float32, shape=[None, len(LABELS)],
                            name='label')

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

    keep_prob = tf.placeholder(tf.float32, name='keep_probability')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, len(LABELS)])
    b_fc2 = bias_variable([len(LABELS)])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.identity(y_conv, name='labels_softmax')

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                       name='train_step')
    prediction = tf.argmax(y_conv, 1, name='predict')

    return {
        'train_step': train_step,
        'x': x,
        'y': y_,
        's': s,
        'keep_prob': keep_prob,
        'y_predict': prediction,
    }


def log_mel_spectrogram_cnn():
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    with tf.variable_scope('training'):
        x = tf.placeholder(tf.float32, shape=[None, SAMPLE_INPUT_LENGTH],
                           name='wav_input')
        s = tf.placeholder(tf.int32, shape=[None, 1],
                           name='sample_rate')
        y_ = tf.placeholder(tf.float32, shape=[None, len(LABELS)],
                            name='label')

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

    image_size = [log_mel_spectrograms.shape[-2].value,
                  log_mel_spectrograms.shape[-1].value]
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    log_mel_images = tf.reshape(log_mel_spectrograms, [-1] + image_size + [1])
    h_conv1 = tf.nn.relu(conv2d(log_mel_images, W_conv1) + b_conv1)
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

    keep_prob = tf.placeholder(tf.float32, name='keep_probability')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, len(LABELS)])
    b_fc2 = bias_variable([len(LABELS)])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.identity(y_conv, name='labels_softmax')

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                       name='train_step')
    prediction = tf.argmax(y_conv, 1, name='predict')

    return {
        'train_step': train_step,
        'x': x,
        'y': y_,
        's': s,
        'keep_prob': keep_prob,
        'y_predict': prediction,
    }


# TODO: Need to understand what the resnet_size and num_mel_bins parameters
# does
def log_mel_spectrogram_resnet(resnet_size, batch_size,
                               num_training_samples):
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    with tf.variable_scope('training'):
        x = tf.placeholder(tf.float32, shape=[None, SAMPLE_INPUT_LENGTH],
                           name='wav_input')
        s = tf.placeholder(tf.int32, shape=[None, 1],
                           name='sample_rate')
        y_ = tf.placeholder(tf.float32, shape=[None, len(LABELS)],
                            name='label')
        is_training = tf.placeholder(tf.bool, name='is_training')
    global_step = tf.train.get_or_create_global_step()

    # MFCC related code stolen from
    # https://tensorflow.org/api_guides/python/contrib.signal#Computing_spectrograms
    stfts = tf.contrib.signal.stft(x, frame_length=128, frame_step=64,
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

    image_size = [log_mel_spectrograms.shape[-2].value,
                  log_mel_spectrograms.shape[-1].value]
    log_mel_channels = tf.reshape(
        log_mel_spectrograms,
        [-1, image_size[0], image_size[1], 1])
    log_mel_channels = tf.image.resize_images(
        log_mel_channels,
        size=(32, 32))

    # Much of what is below is copied from imagenet_main.py (which is from
    # Tensorflow official models
    #
    # Using CIFAR-10 model, data size matches more closely to this situation
    network = resnet_model.cifar10_resnet_v2_generator(
        resnet_size, len(LABELS) + 0)
    logits = network(inputs=log_mel_channels,
                     is_training=is_training)

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='predict'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=y_)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    #
    # Increasing rate to try to help with convergence
    initial_learning_rate = 0.1 * batch_size / 128
    batches_per_epoch = num_training_samples / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    # learning_rate = tf.train.exponential_decay(initial_learning_rate,
    #                                            global_step,
    #                                            batches_per_epoch,
    #                                            0.95)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)
    # optimizer = tf.train.GradientDescentOptimizer(
    #     learning_rate=learning_rate)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    minimize = optimizer.minimize(loss, global_step, name='train_step')
    with tf.control_dependencies(update_ops):
        train_op = tf.cond(
            is_training,
            lambda: minimize,
            lambda: tf.no_op(),
        )

    with tf.name_scope('train_metrics'):
        correct_prediction = tf.equal(predictions['classes'], tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')
        # Create a tensor named train_accuracy for logging purposes.
        tf.summary.scalar('train_accuracy', accuracy)

    return {
        'train_step': train_op,
        'x': x,
        'y': y_,
        's': s,
        'is_training': is_training,
        'y_predict': predictions['classes'],
    }
