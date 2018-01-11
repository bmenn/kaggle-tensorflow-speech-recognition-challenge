'''Test tfspeech.models module

'''
import numpy as np
import pytest
import tensorflow as tf

import tfspeech.models as models


@pytest.mark.parametrize(
    'model_class,model_config',
    [
        (models.mfcc_spectrogram_cnn, {}),
        (models.log_mel_spectrogram_cnn, {}),
        (models.log_mel_spectrogram_resnet,
         {'num_training_samples': 5, 'batch_size': 5, 'resnet_size': 20}),
        (models.log_mel_spectrogram_resnet_custom,
         {'num_training_samples': 5, 'batch_size': 5, 'resnet_size': 56}),
    ]
)
def test_mfcc_spectrogram_cnn(model_class, model_config):
    '''Confirm that model is actually updating/training given data to catch
    embrassingly simple mistakes

    '''
    data = {
        'x': np.eye(5, 16000),
        'y': np.eye(5, 12),
        's': 16000 * np.ones(shape=(5, 1)),
        'keep_prob': 1.0,
    }

    feed_dict = {
        'training/wav_input:0': data['x'],
        'training/label:0': data['y'],
        'training/sample_rate:0': data['s'],
    }


    with tf.Session(graph=tf.Graph()) as sess:
        ops = model_class(**model_config)
        sess.run(tf.global_variables_initializer())

        try:
            sess.graph.get_operation_by_name('keep_probability')
            feed_dict.update({'keep_probability:0': data['keep_prob']})
        except KeyError:
            pass
        try:
            sess.graph.get_operation_by_name('training/is_training')
            feed_dict.update({'training/is_training:0': False})
        except KeyError:
            pass

        prediction_t0 = sess.run('predict:0',
                                 feed_dict=feed_dict)

        # Normally training is not done with a dropout set to 0 (i.e.,
        # keep_prob=1.0), but we only need to check that the model is
        # updating.
        try:
            sess.graph.get_operation_by_name('training/is_training')
            feed_dict.update({'training/is_training:0': True})
        except KeyError:
            pass
        for _ in range(1000):
            sess.run('train_step', feed_dict=feed_dict)

        try:
            sess.graph.get_operation_by_name('training/is_training')
            feed_dict.update({'training/is_training:0': False})
        except KeyError:
            pass
        prediction_final = sess.run('predict:0',
                                    feed_dict=feed_dict)

        assert not np.allclose(prediction_t0, prediction_final)
