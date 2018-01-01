'''Test tfspeech.models module

'''
import numpy as np
import tensorflow as tf

import tfspeech.models as models


def test_mfcc_spectrogram_cnn():
    '''Confirm that model is actually updating/training given data to catch
    embrassingly simple mistakes

    '''
    data = {
        'x': np.random.rand(5, 16000),
        'y': np.eye(5, 12),
        's': 16000 * np.ones(shape=(5, 1)),
        'keep_prob': 1.0,
    }
    ops = models.mfcc_spectrogram_cnn()

    feed_dict = {
        ops['x']: data['x'],
        ops['y']: data['y'],
        ops['s']: data['s'],
        ops['keep_prob']: data['keep_prob']
    }

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prediction_t0 = ops['y_predict'].eval(feed_dict=feed_dict)

        # Normally training is not done with a dropout set to 0 (i.e.,
        # keep_prob=1.0), but we only need to check that the model is
        # updating.
        for _ in range(10):
            ops['train_step'].run(feed_dict=feed_dict)

        prediction_final = ops['y_predict'].eval(feed_dict=feed_dict)

        assert not np.allclose(prediction_t0, prediction_final)
