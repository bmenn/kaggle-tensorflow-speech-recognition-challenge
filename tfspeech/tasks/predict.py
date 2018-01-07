'''Model inference/benchmarking

'''
import os

import h5py
import luigi
import numpy as np
import tensorflow as tf

import tfspeech.tasks.data as tasks_data
import tfspeech.models as models


class ModelFiles(luigi.ExternalTask):

    model_path = luigi.Parameter()

    def output(self):
        model_pb_path = '/'.join([self.model_path, 'saved_model.pb'])
        return luigi.LocalTarget(model_pb_path),


@luigi.util.inherits(ModelFiles)
class GenerateModelTestPredictions(luigi.Task):

    '''Generates models predicts for test data

    '''

    resources = {'tensorflow': 1}

    def requires(self):
        return {
            'model': self.clone(ModelFiles),
            'test_data': tasks_data.ConvertTestWavToArray()
        }

    @property
    def model_id(self):
        return os.path.split(self.model_path)[1]

    def output(self):
        return luigi.LocalTarget(
           os.path.join('data/uploads/', self.model_id + '_predictions.csv')
        )

    def run(self):
        with h5py.File(self.input()['test_data']['data'].path, 'r') as hf:
            x = hf['data'][:]

        with tf.Session() as sess:
            tf.saved_model.loader.load(
                sess,
                ['model', self.model_id],
                self.model_path
            )

            # Normally training is not done with a dropout set to 0 (i.e.,
            # keep_prob=1.0), but we only need to check that the model is
            # updating.
            predictions = []
            for i in range(0, len(x), 1024):
                feed_dict = {
                    'training/wav_input:0': x[i:i+1024],
                    'training/sample_rate:0': 16000 * np.ones((len(x[i:i+1024]), 1)),
                    'keep_probability:0': 1,
                }
                predictions.append(sess.run(
                    'predict:0',
                    feed_dict=feed_dict
                ))

        predictions = np.concatenate(predictions).tolist()
        with self.input()['test_data']['files'].open('r') as f:
            filenames = [fname.strip() for fname in f.readlines()]
        with self.output().open('w') as f:
            f.write('fname,label\n')
            f.write('\n'.join(
                ','.join((fname, models.LABELS[p]))
                for fname, p in zip(filenames, predictions)
            ))
