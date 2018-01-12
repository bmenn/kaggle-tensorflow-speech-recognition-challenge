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

    log_mel_spectrograms = log_mel_spectrogram(x, s)
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

    log_mel_spectrograms = log_mel_spectrogram(x, s)

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


def log_mel_spectrogram_resnet(resnet_size, batch_size,
                               num_training_samples):
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    inputs = input_tensors()
    x = inputs['wav_input']
    s = inputs['sample_rate']
    y_ = inputs['label']
    is_training = inputs['is_training']
    global_step = tf.train.get_or_create_global_step()

    spectrogram_opts = {'frame_length': 128,
                        'frame_step': 64,
                        'fft_length':1024,
                        'lower_hertz': 80.0,
                        'upper_hertz': 7600.0,
                        'num_mel_bins': 64}
    log_mel_spectrograms = log_mel_spectrogram(x, s, **spectrogram_opts)

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
    initial_learning_rate = 0.1 * batch_size / 128
    batches_per_epoch = num_training_samples / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [50, 75, 100]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

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


def log_mel_spectrogram_resnet_custom(
        block_sizes, filters, batch_size, num_training_samples,
        spectrogram_opts=None):
    # TODO: Add inference graph, see
    # tensorflow/tensorflow/examples/speech_commands/freeze.py
    if spectrogram_opts is None:
        spectrogram_opts = {}
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    inputs = input_tensors()
    x = inputs['wav_input']
    s = inputs['sample_rate']
    y_ = inputs['label']
    is_training = inputs['is_training']
    global_step = tf.train.get_or_create_global_step()

    log_mel_spectrograms = log_mel_spectrogram(x, s, **spectrogram_opts)

    # For a 16000 sample and default settings, size=98x40
    image_size = [log_mel_spectrograms.shape[-2].value,
                  log_mel_spectrograms.shape[-1].value]
    log_mel_channels = tf.reshape(
        log_mel_spectrograms,
        [-1, image_size[0], image_size[1], 1])

    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      log_mel_channels = tf.transpose(log_mel_channels, [0, 3, 1, 2])

    conv1 = resnet_model.conv2d_fixed_padding(
        inputs=log_mel_channels,
        filters=64,
        kernel_size=[20, 8],
        strides=1,
        data_format=data_format
        )

    inputs = conv1
    for i in range(len(block_sizes)):
        # TODO Maybe add pooling after each block?
        inputs = resnet_model.block_layer(
            inputs=inputs, filters=filters[i],
            block_fn=resnet_model.building_block, blocks=block_sizes[i],
            strides=1, is_training=is_training,
            name='block_layer%d' % (i + 1),
            data_format=data_format,
        )
    inputs = resnet_model.batch_norm_relu(inputs, is_training,
                                          data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=2, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    # TODO Consider dropout
    inputs = tf.reshape(inputs,
                        [-1,
                         inputs.shape[-3].value
                         * inputs.shape[-2].value
                         * inputs.shape[-1].value])
    logits = tf.layers.dense(inputs=inputs, units=len(LABELS))
    logits = tf.identity(logits, 'final_dense')

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
    initial_learning_rate = 0.1 * batch_size / 128
    batches_per_epoch = num_training_samples / batch_size

    boundaries = [int(batches_per_epoch * epoch)
                  for epoch in [50, 75, 100]]
    values = [initial_learning_rate * decay
              for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

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
