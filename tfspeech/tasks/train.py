'''Train models

'''
import hashlib
import json
import logging
import os

import h5py
import luigi
import numpy as np
import tensorflow as tf

import tfspeech.models as models
import tfspeech.tasks.data as data

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_sh = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)-15s %(message)s')
_sh.setFormatter(_formatter)
LOGGER.addHandler(_sh)

class TrainTensorflowModel(luigi.Task):

    '''Abstract base class for training Tensorflow models

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
    def model_class():
        '''Returns callable object that is a model

        Should be implemented as a static method
        '''
        raise NotImplementedError

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

    @property
    def checkpoint_path(self):
        return self.save_path + '.ckpt'

    def build_graph(self, num_samples):
        return self.model_class()()

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

        data_indices = np.arange(len(y))
        np.random.shuffle(data_indices)
        start_index = 0

        ops = self.build_graph(len(y))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('tflogs/' + self.model_id)
        steps = int(self.num_epochs * len(y) / self.batch_size)
        # TODO: Need to implement some kind of atomic like behavior similar
        # to Target classes. Otherwise, if the save path exists the builder
        # with fail to instantiate.
        saver = tf.train.Saver()
        initial_step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.split(self.checkpoint_path)[0])
            # FIXME This checkpoint check only works if only one model
            # (which also the current one) has been checkpointed
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, self.checkpoint_path)
                global_step = tf.train.get_or_create_global_step()
                initial_step = sess.run(global_step)

            try:
                for i in range(initial_step, steps):
                    if start_index + self.batch_size > len(y):
                        np.random.shuffle(data_indices)
                        start_index = 0
                    stop_index = start_index + self.batch_size
                    feed_dict = {
                        'training/wav_input:0': x[start_index:stop_index],
                        'training/label:0': y[start_index:stop_index],
                        'training/sample_rate:0': 16000 * np.ones((self.batch_size, 1)),
                    }
                    try:
                        sess.graph.get_operation_by_name('keep_probability')
                        feed_dict.update({'keep_probability:0': 1.0 - self.dropout_rate})
                    except KeyError:
                        pass
                    try:
                        sess.graph.get_operation_by_name('is_training')
                        feed_dict.update({'is_training:0': True})
                    except KeyError:
                        pass

                    if i % 10 == 0:
                        summary, _, __ = sess.run(
                            [merged, 'train_step', 'train_metrics/accuracy'],
                            feed_dict=feed_dict
                        )
                        writer.add_summary(summary, i)
                    else:
                        sess.run(
                            'train_step',
                            feed_dict=feed_dict
                        )
                        start_index += self.batch_size

                    if i % 1000 == 0:
                        saver.save(sess, self.save_path + '.ckpt')
                        LOGGER.info('Model %s training progress: %d/%d' %
                                    (self.model_id, i, steps))
            except KeyboardInterrupt as e:
                LOGGER.info('Stopping model %s, training progress: %d/%d' %
                            (self.model_id, i, steps))
                saver.save(sess, self.save_path + '.ckpt')
                writer.flush()
                raise e


            builder = tf.saved_model.builder.SavedModelBuilder(self.save_path)
            builder.add_meta_graph_and_variables(sess, ['model',
                                                        self.model_id])

        builder.save()
        param_json = json.dumps(self.to_str_params(only_significant=True),
                                separators=(',', ':'), sort_keys=True)
        with self.output()['metadata'].open('w') as f:
            f.write(param_json)


class TrainParametrizedTensorflowModel(TrainTensorflowModel):

    '''Abstract base class for training parametrized Tensorflow models

    '''

    model_settings = luigi.DictParameter()

    def build_graph(self, num_samples):
        return self.model_class()(**self.model_settings)


class MfccSpectrogramCNN(TrainTensorflowModel):

    '''Trains a CNN using MFCC spectrograms

    '''

    @staticmethod
    def model_class():
        return models.mfcc_spectrogram_cnn


class LogMelSpectrogramCNN(TrainTensorflowModel):

    '''Trains a CNN using log Mel spectrograms

    '''

    @staticmethod
    def model_class():
        return models.log_mel_spectrogram_cnn


class LogMelSpectrogramResNet(TrainParametrizedTensorflowModel):

    '''Trains a ResNet using log Mel spectrograms

    '''
    batch_size = luigi.IntParameter(default=256)
    num_epochs = luigi.IntParameter(default=120)

    @staticmethod
    def model_class():
        return models.log_mel_spectrogram_resnet

    def build_graph(self, num_samples):
        data_sizes = {
            'num_training_samples': num_samples,
            'batch_size': self.batch_size,
        }
        settings = {**self.model_settings, **data_sizes}
        return self.model_class()(**settings)


@luigi.util.inherits(TrainTensorflowModel)
class ValidateTensorflowModel(luigi.Task):

    '''Validates TrainTensorflowModel

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
                }
                try:
                    sess.graph.get_operation_by_name('keep_probability')
                    feed_dict.update({'keep_probability:0': 1.0})
                except KeyError:
                    pass
                try:
                    sess.graph.get_operation_by_name('is_training')
                    feed_dict.update({'is_training:0': False})
                except KeyError:
                    pass
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


@luigi.util.requires(MfccSpectrogramCNN)
class ValidateMfccSpectrogramCNN(ValidateTensorflowModel):

    '''Validates MfccSpectrogramCNN

    '''
    pass


@luigi.util.requires(LogMelSpectrogramCNN)
class ValidateLogMelSpectrogramCNN(ValidateTensorflowModel):

    '''Validates LogMelSpectrogramCNN

    '''
    pass


@luigi.util.requires(LogMelSpectrogramResNet)
class ValidateLogMelSpectrogramResNet(ValidateTensorflowModel):

    '''Validates LogMelSpectrogramResNet

    '''

    batch_size = luigi.IntParameter(default=256)
    num_epochs = luigi.IntParameter(default=120)


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

    validation_tasks = [ValidateMfccSpectrogramCNN,
                        ValidateLogMelSpectrogramCNN]
    train_all_data_task = [MfccSpectrogramCNN,
                           LogMelSpectrogramCNN,
                           LogMelSpectrogramResNet]
    # Some models are not configurable due to old implementation. In an
    # effort to not delete previously trained data, None is use to hack
    # older implementations into compatibility.
    model_settings = [
        None,
        None,
        {'resnet_size': 18}
    ]

    def model_tasks(self):
        models = []
        for i in range(len(self.data_files)):
            data_subset = self.data_files[:i] + self.data_files[i+1:]
            labels_subset = self.label_files[:i] + self.label_files[i+1:]
            for j in range(len(data_subset)):
                for task_class, settings in zip(self.validation_tasks,
                                                self.model_settings):
                    if isinstance(task_class,
                                  ValidateLogMelSpectrogramResNet):
                        # Resnet might take a while to training, so only
                        # doing one set
                        continue

                    if settings is None:
                        models.append(task_class(
                            data_files=data_subset[:j+1],
                            label_files=labels_subset[:j+1],
                            validation_data=[self.data_files[i]],
                            validation_labels=[self.label_files[i]],
                        ))
                    else:
                        models.append(task_class(
                            data_files=data_subset[:j+1],
                            label_files=labels_subset[:j+1],
                            validation_data=[self.data_files[i]],
                            validation_labels=[self.label_files[i]],
                            model_settings=settings,
                        ))
            # Add Resnet model
            models.append(ValidateLogMelSpectrogramResNet(
                data_files=data_subset,
                label_files=labels_subset,
                validation_data=[self.data_files[i]],
                validation_labels=[self.label_files[i]],
                model_settings=self.model_settings[2]
            ))

        for task_class, settings in zip(self.train_all_data_task,
                                        self.model_settings):
            if settings is None:
                models.append(task_class(
                    data_files=self.data_files,
                    label_files=self.label_files,
                ))
            else:
                models.append(task_class(
                    data_files=self.data_files,
                    label_files=self.label_files,
                    model_settings=settings,
                ))

        return models

    def output(self):
        return {
            (
                task.requires().model_id
                if isinstance(task, ValidateTensorflowModel)
                else task.model_id
            ): (
                {**task.output(), **task.input()}
                if isinstance(task, ValidateTensorflowModel)
                else {**task.output()}
            )
            for task in self.model_tasks()
        }

    def run(self):
        yield self.model_tasks()
