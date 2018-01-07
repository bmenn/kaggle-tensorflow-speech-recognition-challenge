'''Test tfspeech.models module

'''
import numpy as np
import pytest
import tensorflow as tf

import tfspeech.models as models


@pytest.mark.parametrize(
    'model_class',
    [
        models.mfcc_spectrogram_cnn,
        models.log_mel_spectrogram_cnn,
    ]
)
def test_mfcc_spectrogram_cnn(model_class):
    '''Confirm that model is actually updating/training given data to catch
    embrassingly simple mistakes

    '''
    data = {
        'x': np.random.rand(5, 16000),
        'y': np.eye(5, 12),
        's': 16000 * np.ones(shape=(5, 1)),
        'keep_prob': 1.0,
    }
    ops = model_class()

    feed_dict = {
        'training/wav_input:0': data['x'],
        'training/label:0': data['y'],
        'training/sample_rate:0': data['s'],
        'keep_probability:0': data['keep_prob']
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prediction_t0 = sess.run('predict:0',
                                 feed_dict=feed_dict)

        # Normally training is not done with a dropout set to 0 (i.e.,
        # keep_prob=1.0), but we only need to check that the model is
        # updating.
        for _ in range(10):
            sess.run(['train_step'], feed_dict=feed_dict)

        prediction_final = sess.run('predict:0',
                                    feed_dict=feed_dict)

        assert not np.allclose(prediction_t0, prediction_final)
