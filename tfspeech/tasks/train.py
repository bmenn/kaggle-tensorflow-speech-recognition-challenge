'''Train models

'''
import hashlib
import json
import os

import h5py
import luigi
import numpy as np
import tensorflow as tf

import tfspeech.models as models
import tfspeech.tasks.data as data


@luigi.util.requires(data.DoDataPreProcessing)
class InitiateTraining(luigi.Task):

    '''Initiates training of all models

    This task act a gateway/middle-man between pre-processing and training.
    '''

    def run(self):
        yield TrainAllModels(
            data_files=[target.path for target in self.input()['data']],
            label_files=[target.path for target in self.input()['labels']]
        )


class TrainAllModels(luigi.Task):
    data_files = luigi.ListParameter()
    label_files = luigi.ListParameter()

    def model_tasks(self):
        mfcc_spectrogram_cnn_models = []
        for i in range(len(self.data_files)):
            data_subset = self.data_files[:i] + self.data_files[i+1:]
            labels_subset = self.label_files[:i] + self.label_files[i+1:]
            for j in range(len(data_subset)):
                model = ValidateMfccSpectrogramCNN(
                    data_files=data_subset[:j+1],
                    label_files=labels_subset[:j+1],
                    validation_data=[self.data_files[i]],
                    validation_labels=[self.label_files[i]],
                )
                mfcc_spectrogram_cnn_models.append(model)

        return mfcc_spectrogram_cnn_models

    def output(self):
        return {task.requires().model_id: {**task.output(), **task.input()}
                for task in self.model_tasks()}

    def run(self):
        yield self.model_tasks()


class MfccSpectrogramCNN(luigi.Task):

    '''Trains a CNN using MFCC spectrograms

    `data_files` and `label_files` must aligned such that the order of data
    matches the order of labels.

    '''
    data_files = luigi.ListParameter()
    label_files = luigi.ListParameter()
    downsample_rate = luigi.FloatParameter(default=1.0)
    num_epochs = luigi.IntParameter(default=40)
    batch_size = luigi.IntParameter(default=32)
    dropout_rate = luigi.FloatParameter(default=0.2)

    resources = {'tensorflow': 1}

    def output(self):
        model_pb_path = '/'.join([self.save_path, 'saved_model.pb'])
        metadata_path = '/'.join([self.save_path, 'METADATA.json'])
        return {
            'pb': luigi.LocalTarget(model_pb_path),
            'metadata': luigi.LocalTarget(metadata_path)
        }

    @property
    def model_id(self):
        param_str = json.dumps(self.to_str_params(only_significant=True),
                               separators=(',', ':'), sort_keys=True)
        param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()

        return '_'.join([
            self.get_task_family(),
            param_hash
        ])

    @property
    def save_path(self):
        return 'data/models/' + self.model_id

    def run(self):
        x = []
        for data_file in self.data_files:
            with h5py.File(data_file, 'r') as hf:
                x.append(hf['data'][:])
        x = np.vstack(x)

        y = []
        for label_file in self.label_files:
            with h5py.File(label_file, 'r') as hf:
                y.append(hf['data'][:])
        y = np.vstack(y)

        ops = models.mfcc_spectrogram_cnn()

        data_indices = np.arange(len(y))
        np.random.shuffle(data_indices)
        start_index = 0

        steps = int(self.num_epochs * len(y) / self.batch_size)
        # TODO: Need to implement some kind of atomic like behavior similar
        # to Target classes. Otherwise, if the save path exists the builder
        # with fail to instantiate.
        builder = tf.saved_model.builder.SavedModelBuilder(self.save_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Normally training is not done with a dropout set to 0 (i.e.,
            # keep_prob=1.0), but we only need to check that the model is
            # updating.
            for _ in range(steps):
                if start_index + self.batch_size > len(y):
                    np.random.shuffle(data_indices)
                    start_index = 0
                stop_index = start_index + self.batch_size
                feed_dict = {
                    'training/wav_input:0': x[start_index:stop_index],
                    'training/label:0': y[start_index:stop_index],
                    'training/sample_rate:0': 16000 * np.ones((self.batch_size, 1)),
                    'keep_probability:0': 1 - self.dropout_rate,
                }
                sess.run(
                    'train_step',
                    feed_dict=feed_dict
                )
                start_index += self.batch_size

            builder.add_meta_graph_and_variables(sess, ['model',
                                                        self.model_id])

        builder.save()
        param_json = json.dumps(self.to_str_params(only_significant=True),
                                separators=(',', ':'), sort_keys=True)
        with self.output()['metadata'].open('w') as f:
            f.write(param_json)


@luigi.util.requires(MfccSpectrogramCNN)
class ValidateMfccSpectrogramCNN(luigi.Task):

    '''Validates MfccSpectrogramCNN

    '''
    validation_data = luigi.ListParameter()
    validation_labels = luigi.ListParameter()

    resources = {'tensorflow': 1}

    def output(self):
        base_path = self.requires().save_path
        return {
            'predictions': luigi.LocalTarget(os.path.join(base_path,
                                                         'predictions.csv')),
            'metrics': luigi.LocalTarget(os.path.join(base_path,
                                                      'metrics.json')),
        }

    def run(self):
        x = []
        for data_file in self.validation_data:
            with h5py.File(data_file, 'r') as hf:
                x.append(hf['data'][:])
        x = np.vstack(x)

        y = []
        for label_file in self.validation_labels:
            with h5py.File(label_file, 'r') as hf:
                y.append(hf['data'][:])
        y = np.vstack(y)


        with tf.Session() as sess:
            tf.saved_model.loader.load(
                sess,
                ['model', self.requires().model_id],
                self.requires().save_path
            )

            # Normally training is not done with a dropout set to 0 (i.e.,
            # keep_prob=1.0), but we only need to check that the model is
            # updating.
            predictions = []
            for i in range(0, len(y), 1024):
                feed_dict = {
                    'training/wav_input:0': x[i:i+1024],
                    'training/label:0': y[i:i+1024],
                    'training/sample_rate:0': 16000 * np.ones((len(y[i:i+1024]), 1)),
                    'keep_probability:0': 1,
                }
                predictions.append(sess.run(
                    'predict:0',
                    feed_dict=feed_dict
                ))

        predictions = np.concatenate(predictions).reshape((-1, 1))
        y_labels = np.argmax(y, axis=1).reshape((-1, 1))
        prediction_labels = np.hstack([predictions, y_labels])
        with h5py.File(self.output()['predictions'].path, 'w') as hf:
            hf.create_dataset('data', data=prediction_labels)

        metrics = {
            'accuracy': np.sum(np.equal(predictions, y_labels)) / len(y_labels)
        }
        metrics_json = json.dumps(metrics, separators=(',', ':'),
                                  sort_keys=True)
        with self.output()['metrics'].open('w') as f:
            f.write(metrics_json)
