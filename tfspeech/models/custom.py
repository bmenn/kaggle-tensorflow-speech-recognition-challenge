import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

from .resnet import resnet_model as resnet_model
# from tfspeech.models import (LABELS, SAMPLE_RATE, SAMPLE_INPUT_LENGTH, _MOMENTUM,
#                _WEIGHT_DECAY, log_mel_spectrogram)
from .utils import input_tensors, log_mel_spectrogram
from .constants import *
from .constants import _WEIGHT_DECAY, _MOMENTUM


def spectrogram_resnet_block(x, s, spectrogram_opts,
                             block_sizes, block_strides, filters,
                             kernel_sizes, data_format, is_training,
                             final_pool_type, final_pool_size):
    log_mel_spectrograms = log_mel_spectrogram(x, s, **spectrogram_opts)

    # For a 16000 sample and default settings, size=98x40
    image_size = [log_mel_spectrograms.shape[-2].value,
                  log_mel_spectrograms.shape[-1].value]
    log_mel_channels = tf.reshape(
        log_mel_spectrograms,
        [-1, image_size[0], image_size[1], 1])
    # log_mel_channels = log_mel_channels - tf.reduce_mean(log_mel_channels)

    if data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        log_mel_channels = tf.transpose(log_mel_channels, [0, 3, 1, 2])

    inputs = log_mel_channels
    for i in range(len(filters)):
        # TODO Maybe add pooling after each block?
        inputs = resnet_model.block_layer(
            inputs=inputs, filters=filters[i],
            block_fn=resnet_model.building_block,
            blocks=block_sizes[i],
            kernel_size=kernel_sizes[i],
            strides=block_strides[i], is_training=is_training,
            name='block_layer%d' % (i + 1),
            data_format=data_format,
        )
    inputs = resnet_model.batch_norm_relu(inputs, is_training,
                                          data_format)
    if final_pool_type is not None:
        inputs = final_pool_type(
            inputs=inputs, pool_size=final_pool_size, strides=1, padding='VALID',
            data_format=data_format)

    return inputs


def log_mel_spectrogram_resnet_v2(
        block_sizes, block_strides, filters, kernel_sizes,
        batch_size, num_training_samples, spectrogram_opts=None,
        dropout_rate=0.0, initial_learning_rate=0.01,
        lr_decay_rate=0.5, lr_decay_epochs=10,
        final_pool_size=8, final_pool_type=None):
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    if spectrogram_opts is None:
        spectrogram_opts = {}

    if final_pool_type is None or final_pool_type == 'avg':
        final_pool_type = tf.layers.average_pooling2d
    elif final_pool_type == 'max':
        final_pool_type = tf.layers.max_pooling2d
    elif final_pool_type == 'no_pooling':
        final_pool_type = None
    else:
        raise ValueError

    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    inputs = input_tensors()
    x = inputs['wav_input']
    s = inputs['sample_rate']
    y_ = inputs['label']
    is_training = inputs['is_training']
    keep_prob = tf.placeholder(tf.float32, name='keep_probability')
    global_step = tf.train.get_or_create_global_step()

    inputs = spectrogram_resnet_block(x, s, spectrogram_opts,
                                      block_sizes, block_strides,
                                      filters, kernel_sizes,
                                      data_format, is_training,
                                      final_pool_type, final_pool_size )

    inputs = tf.reshape(inputs,
                        [-1,
                         inputs.shape[-3].value
                         * inputs.shape[-2].value
                         * inputs.shape[-1].value])
    inputs = resnet_model.batch_norm_relu(
        tf.layers.dense(inputs=inputs, units=1024),
        is_training, data_format)
    logits = tf.layers.dense(inputs=inputs, units=len(LABELS))
    logits = tf.identity(logits, 'final_dense')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='predict'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=y_)

    # Create a tensor named cross_entropy for logging purposes.
    with tf.variable_scope('loss'):
        tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    initial_learning_rate = initial_learning_rate
    batches_per_epoch = num_training_samples / batch_size

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step,
        decay_steps=lr_decay_epochs * batches_per_epoch,
        decay_rate=lr_decay_rate,
        staircase=True)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, name='train_step')

    with tf.name_scope('train_metrics'):
        correct_prediction = tf.equal(predictions['classes'], tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')

    return {
        'train_step': train_op,
        'x': x,
        'y': y_,
        's': s,
        'is_training': is_training,
        'y_predict': predictions['classes'],
    }


def mfcc_spectrogram_resnet(
        block_sizes, block_strides, filters, kernel_sizes,
        batch_size, num_training_samples, spectrogram_opts=None,
        dropout_rate=0.0, initial_learning_rate=0.01,
        final_pool_size=8, final_pool_type=None):
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    if spectrogram_opts is None:
        spectrogram_opts = {}

    if final_pool_type is None or final_pool_type == 'avg':
        final_pool_type = tf.layers.average_pooling2d
    elif final_pool_type == 'max':
        final_pool_type = tf.layers.max_pooling2d
    elif final_pool_type == 'no_pooling':
        final_pool_type = None
    else:
        raise ValueError

    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    inputs = input_tensors()
    x = inputs['wav_input']
    s = inputs['sample_rate']
    y_ = inputs['label']
    is_training = inputs['is_training']
    keep_prob = tf.placeholder(tf.float32, name='keep_probability')
    global_step = tf.train.get_or_create_global_step()

    log_mel_spectrograms = log_mel_spectrogram(x, s, **spectrogram_opts)
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., 1:13]

    # For a 16000 sample and default settings, size=98x40
    image_size = [mfccs.shape[-2].value,
                  mfccs.shape[-1].value]
    mfccs = tf.reshape(
        mfccs,
        [-1, image_size[0], image_size[1], 1])
    mfccs = mfccs - tf.reduce_mean(mfccs)

    if data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        mfccs = tf.transpose(mfccs, [0, 3, 1, 2])

    inputs = mfccs
    for i in range(len(filters)):
        # TODO Maybe add pooling after each block?
        inputs = resnet_model.block_layer(
            inputs=inputs, filters=filters[i],
            block_fn=resnet_model.building_block,
            blocks=block_sizes[i],
            kernel_size=kernel_sizes[i],
            strides=block_strides[i], is_training=is_training,
            name='block_layer%d' % (i + 1),
            data_format=data_format,
        )
    inputs = resnet_model.batch_norm_relu(inputs, is_training,
                                          data_format)
    if final_pool_type is not None:
        inputs = final_pool_type(
            inputs=inputs, pool_size=final_pool_size, strides=1, padding='VALID',
            data_format=data_format)

    inputs = tf.reshape(inputs,
                        [-1,
                         inputs.shape[-3].value
                         * inputs.shape[-2].value
                         * inputs.shape[-1].value])
    inputs = tf.nn.dropout(inputs, keep_prob)
    logits = tf.layers.dense(inputs=inputs, units=len(LABELS))
    logits = tf.identity(logits, 'final_dense')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='predict'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=y_)

    # Create a tensor named cross_entropy for logging purposes.
    with tf.variable_scope('loss'):
        tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    initial_learning_rate = initial_learning_rate
    batches_per_epoch = num_training_samples / batch_size

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step,
        decay_steps=10 * batches_per_epoch, decay_rate=0.5,
        staircase=True)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, name='train_step')

    with tf.name_scope('train_metrics'):
        correct_prediction = tf.equal(predictions['classes'], tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')

    return {
        'train_step': train_op,
        'x': x,
        'y': y_,
        's': s,
        'is_training': is_training,
        'y_predict': predictions['classes'],
    }


def multi_spectrogram_resnet(
        block_sizes, block_strides, filters, kernel_sizes,
        batch_size, num_training_samples, spectrogram_opts,
        initial_learning_rate=0.01,
        lr_decay_rate=0.5, lr_decay_epochs=10,
        final_pool_size=8, final_pool_type=None):
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    if spectrogram_opts is None:
        spectrogram_opts = {}

    if final_pool_type is None or final_pool_type == 'avg':
        final_pool_type = tf.layers.average_pooling2d
    elif final_pool_type == 'max':
        final_pool_type = tf.layers.max_pooling2d
    elif final_pool_type == 'no_pooling':
        final_pool_type = None
    else:
        raise ValueError

    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    inputs = input_tensors()
    x = inputs['wav_input']
    s = inputs['sample_rate']
    y_ = inputs['label']
    is_training = inputs['is_training']
    keep_prob = tf.placeholder(tf.float32, name='keep_probability')
    global_step = tf.train.get_or_create_global_step()

    fine_spectrogram_opts = dict(spectrogram_opts)

    fine_spectrogram_opts.update({
        'frame_step': int(spectrogram_opts['frame_step'] / 2),
        'fft_length': int(spectrogram_opts['fft_length'] / 2),
    })

    fine = spectrogram_resnet_block(x, s, fine_spectrogram_opts,
                                    block_sizes, block_strides,
                                    filters, kernel_sizes,
                                    data_format, is_training,
                                    final_pool_type, final_pool_size)
    medium = spectrogram_resnet_block(x, s, spectrogram_opts,
                                      block_sizes, block_strides,
                                      filters, kernel_sizes,
                                      data_format, is_training,
                                      final_pool_type, final_pool_size)

    fine = tf.reshape(fine,
                        [-1,
                         fine.shape[-3].value
                         * fine.shape[-2].value
                         * fine.shape[-1].value])
    medium = tf.reshape(medium,
                        [-1,
                         medium.shape[-3].value
                         * medium.shape[-2].value
                         * medium.shape[-1].value])

    inputs = tf.concat([fine, medium], 1)
    logits = tf.layers.dense(inputs=inputs, units=len(LABELS))
    logits = tf.identity(logits, 'final_dense')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='predict'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=y_)

    # Create a tensor named cross_entropy for logging purposes.
    with tf.variable_scope('loss'):
        tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    initial_learning_rate = initial_learning_rate
    batches_per_epoch = num_training_samples / batch_size

    boundaries = [int(batches_per_epoch * epoch) for epoch in [25, 50, 60]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.005, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # learning_rate = tf.train.exponential_decay(
    #     initial_learning_rate, global_step,
    #     decay_steps=lr_decay_epochs * batches_per_epoch,
    #     decay_rate=lr_decay_rate,
    #     staircase=True)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, name='train_step')

    with tf.name_scope('train_metrics'):
        correct_prediction = tf.equal(predictions['classes'], tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')

    return {
        'train_step': train_op,
        'x': x,
        'y': y_,
        's': s,
        'is_training': is_training,
        'y_predict': predictions['classes'],
    }


def mel_mfcc_spectrogram_resnet(
        block_sizes, block_strides, filters, kernel_sizes,
        batch_size, num_training_samples, spectrogram_opts,
        initial_learning_rate=0.01,
        lr_decay_rate=0.5, lr_decay_epochs=10,
        final_pool_size=8, final_pool_type=None):
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    if spectrogram_opts is None:
        spectrogram_opts = {}

    if final_pool_type is None or final_pool_type == 'avg':
        final_pool_type = tf.layers.average_pooling2d
    elif final_pool_type == 'max':
        final_pool_type = tf.layers.max_pooling2d
    elif final_pool_type == 'no_pooling':
        final_pool_type = None
    else:
        raise ValueError

    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    inputs = input_tensors()
    x = inputs['wav_input']
    s = inputs['sample_rate']
    y_ = inputs['label']
    is_training = inputs['is_training']
    keep_prob = tf.placeholder(tf.float32, name='keep_probability')
    global_step = tf.train.get_or_create_global_step()

    log_mel = spectrogram_resnet_block(x, s, spectrogram_opts,
                                      block_sizes, block_strides,
                                      filters, kernel_sizes,
                                      data_format, is_training,
                                      final_pool_type, final_pool_size)
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel)[..., 1:13]

    log_mel = tf.reshape(log_mel,
                        [-1,
                         log_mel.shape[-3].value
                         * log_mel.shape[-2].value
                         * log_mel.shape[-1].value])
    mfccs = tf.reshape(mfccs,
                        [-1,
                         mfccs.shape[-3].value
                         * mfccs.shape[-2].value
                         * mfccs.shape[-1].value])

    inputs = tf.concat([log_mel, mfccs], 1)
    logits = tf.layers.dense(inputs=inputs, units=len(LABELS))
    logits = tf.identity(logits, 'final_dense')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='predict'),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=y_)

    # Create a tensor named cross_entropy for logging purposes.
    with tf.variable_scope('loss'):
        tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name])

    initial_learning_rate = initial_learning_rate
    batches_per_epoch = num_training_samples / batch_size

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step,
        decay_steps=lr_decay_epochs * batches_per_epoch,
        decay_rate=lr_decay_rate,
        staircase=True)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, name='train_step')

    with tf.name_scope('train_metrics'):
        correct_prediction = tf.equal(predictions['classes'], tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                  name='accuracy')

    return {
        'train_step': train_op,
        'x': x,
        'y': y_,
        's': s,
        'is_training': is_training,
        'y_predict': predictions['classes'],
    }
