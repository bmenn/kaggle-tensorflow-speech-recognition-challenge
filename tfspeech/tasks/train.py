'''Train models

'''
import hashlib
import itertools
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
    validation_data = luigi.ListParameter()
    validation_labels = luigi.ListParameter()
    downsample_rate = luigi.FloatParameter(default=1.0)
    num_epochs = luigi.IntParameter(default=40)
    batch_size = luigi.IntParameter(default=32)
    dropout_rate = luigi.FloatParameter(default=0.0)

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
        return self.save_path + '_checkpoint/model.ckpt'

    def build_graph(self, num_samples):
        return self.model_class()()

    def _metrics(self, sess, x, y, use_cross_entropy=False):
        if len(x) == 0 or len(y) == 0:
            return None
        feed_dict = {}
        try:
            sess.graph.get_operation_by_name('keep_probability')
            feed_dict.update({'keep_probability:0': 1.0})
        except KeyError:
            pass
        try:
            sess.graph.get_operation_by_name('training/is_training')
            feed_dict['training/is_training:0'] = False
        except KeyError:
            pass

        accuracy = 0
        cross_entropy = 0
        counter = 0
        for i in range(0, len(y), self.batch_size):
            feed_dict.update({
                'training/wav_input:0': x[i:i+self.batch_size],
                'training/label:0': y[i:i+self.batch_size],
                'training/sample_rate:0': 16000 * np.ones((len(y[i:i+self.batch_size]), 1)),
            })
            accuracy += sess.run(
                'train_metrics/accuracy:0',
                feed_dict=feed_dict,
            )
            if use_cross_entropy:
                cross_entropy += sess.run(
                    'loss/cross_entropy:0',
                    feed_dict=feed_dict,
                )
            counter += 1

        accuracy = accuracy / counter
        cross_entropy = cross_entropy / counter
        return {
            'accuracy': accuracy,
            'cross_entropy': cross_entropy if use_cross_entropy else None,
        }

    def _read_data(self, input, data, start, stop):
            data[start:stop, :] = input[:]

    def _combine_h5_files(self, files, dataset='data'):
        # Determine h5 data shape for pre-allocation
        strides = [0]
        widths = []
        for fname in files:
            with h5py.File(fname, 'r') as hf:
                n, w = hf[dataset].shape
                strides.append(strides[-1] + n)
                widths.append(w)

        assert len(set(widths)) == 1, 'Inconsistent data width, giving up'
        data = np.zeros((strides[-1], widths[0]))
        for fname, start, stop in zip(files, strides[:-1], strides[1:]):
            with h5py.File(fname, 'r') as hf:
                # Abuse stack and gc to minimize memory usage
                self._read_data(hf[dataset], data, start, stop)

        return data

    def run(self):
        x = self._combine_h5_files(self.data_files)
        y = self._combine_h5_files(self.label_files)

        x_valid = self._combine_h5_files(self.validation_data)
        y_valid = self._combine_h5_files(self.validation_labels)

        data_indices = np.arange(len(y))
        np.random.shuffle(data_indices)
        start_index = 0

        graph = tf.Graph()
        with graph.as_default():
            ops = self.build_graph(len(y))
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('tflogs/' + self.model_id)
            # TODO: Need to implement some kind of atomic like behavior similar
            # to Target classes. Otherwise, if the save path exists the builder
            # with fail to instantiate.
            saver = tf.train.Saver()
        steps = int(self.num_epochs * len(y) / self.batch_size)
        steps_per_epoch = int(len(y) / self.batch_size)
        initial_step = 0
        with tf.Session(graph=graph) as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.split(self.checkpoint_path)[0])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, self.checkpoint_path)
                global_step = tf.train.get_or_create_global_step()
                initial_step = sess.run(global_step)
            sess.run(tf.global_variables_initializer())

            try:
                for i in range(initial_step, steps):
                    if start_index + self.batch_size > len(y):
                        np.random.shuffle(data_indices)
                        start_index = 0
                    stop_index = start_index + self.batch_size
                    mask = data_indices[start_index:stop_index]
                    feed_dict = {
                        'training/wav_input:0': x[mask],
                        'training/label:0': y[mask],
                        'training/sample_rate:0': 16000 * np.ones((self.batch_size, 1)),
                    }
                    try:
                        sess.graph.get_operation_by_name('keep_probability')
                        feed_dict.update({'keep_probability:0': 1.0 - self.dropout_rate})
                    except KeyError:
                        pass
                    try:
                        sess.graph.get_operation_by_name('training/is_training')
                        feed_dict.update({'training/is_training:0': True})
                    except KeyError:
                        pass
                    if i % steps_per_epoch == 0:
                        saver.save(sess, self.checkpoint_path)
                        LOGGER.info('Model %s training progress: %d/%d' %
                                    (self.model_id, i, steps))

                    if i % int(steps_per_epoch * 0.01) == 0:
                        valid_index = np.random.choice(
                            len(x_valid),
                            size=self.batch_size,
                            replace=False)
                        metrics = self._metrics(
                            sess,
                            x_valid[valid_index], y_valid[valid_index],
                            use_cross_entropy=True)
                        if metrics is not None:
                            for k, v in metrics.items():
                                if v is not None:
                                    summary = tf.Summary()
                                    summary.value.add(
                                        tag='valid_metrics/' + k,
                                        simple_value=v
                                        )
                                    writer.add_summary(summary, i)

                        summary, _, train_accuracy = sess.run(
                            [merged, 'train_step', 'train_metrics/accuracy:0'],
                            feed_dict=feed_dict
                        )
                        train_summary = tf.Summary()
                        train_summary.value.add(
                            tag='train_metrics/accuracy',
                            simple_value=train_accuracy
                        )
                        writer.add_summary(summary, i)
                        writer.add_summary(train_summary, i)
                    else:
                        sess.run(
                            'train_step',
                            feed_dict=feed_dict
                        )
                    start_index += self.batch_size

            except KeyboardInterrupt as e:
                LOGGER.info('Stopping model %s, training progress: %d/%d' %
                            (self.model_id, i, steps))
                saver.save(sess, self.checkpoint_path)
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
    batch_size = luigi.IntParameter()
    num_epochs = luigi.IntParameter()

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


class LogMelSpectrogramResNetCustom(TrainParametrizedTensorflowModel):

    '''Trains a customized ResNet using log Mel spectrograms

    '''
    batch_size = luigi.IntParameter()
    num_epochs = luigi.IntParameter()

    @staticmethod
    def model_class():
        return models.log_mel_spectrogram_resnet_custom

    def build_graph(self, num_samples):
        data_sizes = {
            'num_training_samples': num_samples,
            'batch_size': self.batch_size,
        }
        settings = {**self.model_settings, **data_sizes}
        return self.model_class()(**settings)


class LogMelSpectrogramConvNet(TrainParametrizedTensorflowModel):

    '''Trains a customized ResNet using log Mel spectrograms

    '''
    batch_size = luigi.IntParameter()
    num_epochs = luigi.IntParameter()

    @staticmethod
    def model_class():
        return models.log_mel_spectrogram_convnet

    def build_graph(self, num_samples):
        data_sizes = {
            'num_training_samples': num_samples,
            'batch_size': self.batch_size,
        }
        settings = {**self.model_settings, **data_sizes}
        return self.model_class()(**settings)


class LogMelSpectrogramResNetv2(TrainParametrizedTensorflowModel):

    '''Trains a customized ResNet using log Mel spectrograms

    '''
    batch_size = luigi.IntParameter()
    num_epochs = luigi.IntParameter()

    @staticmethod
    def model_class():
        return models.log_mel_spectrogram_resnet_v2

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

    def _read_data(self, input, data, start, stop):
            data[start:stop, :] = input[:]

    def _combine_h5_files(self, files, dataset='data'):
        # Determine h5 data shape for pre-allocation
        strides = [0]
        widths = []
        for fname in files:
            with h5py.File(fname, 'r') as hf:
                n, w = hf[dataset].shape
                strides.append(strides[-1] + n)
                widths.append(w)

        assert len(set(widths)) == 1, 'Inconsistent data width, giving up'
        data = np.zeros((strides[-1], widths[0]))
        for fname, start, stop in zip(files, strides[:-1], strides[1:]):
            with h5py.File(fname, 'r') as hf:
                # Abuse stack and gc to minimize memory usage
                self._read_data(hf[dataset], data, start, stop)

        return data

    def run(self):
        train_x = self._combine_h5_files(self.data_files)
        train_y = self._combine_h5_files(self.label_files)

        x = self._combine_h5_files(self.validation_data)
        y = self._combine_h5_files(self.validation_labels)

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(
                sess,
                ['model', self.requires().model_id],
                self.requires().save_path
            )

            # Normally training is not done with a dropout set to 0 (i.e.,
            # keep_prob=1.0), but we only need to check that the model is
            # updating.
            predictions = []
            for i in range(0, len(y), self.batch_size):
                feed_dict = {
                    'training/wav_input:0': x[i:i+self.batch_size],
                    'training/label:0': y[i:i+self.batch_size],
                    'training/sample_rate:0': 16000 * np.ones((len(y[i:i+self.batch_size]), 1)),
                }
                try:
                    sess.graph.get_operation_by_name('keep_probability')
                    feed_dict.update({'keep_probability:0': 1.0})
                except KeyError:
                    pass
                try:
                    sess.graph.get_operation_by_name('training/is_training')
                    feed_dict.update({'training/is_training:0': False})
                except KeyError:
                    pass
                predictions.append(sess.run(
                    'predict:0',
                    feed_dict=feed_dict
                ))

            train_predictions = []
            for i in range(0, len(train_y), self.batch_size):
                feed_dict = {
                    'training/wav_input:0': train_x[i:i+self.batch_size],
                    'training/label:0': train_y[i:i+self.batch_size],
                    'training/sample_rate:0': 16000 * np.ones((len(train_y[i:i+self.batch_size]), 1)),
                }
                try:
                    sess.graph.get_operation_by_name('keep_probability')
                    feed_dict.update({'keep_probability:0': 1.0})
                except KeyError:
                    pass
                try:
                    sess.graph.get_operation_by_name('training/is_training')
                    feed_dict.update({'training/is_training:0': False})
                except KeyError:
                    pass
                train_predictions.append(sess.run(
                    'predict:0',
                    feed_dict=feed_dict
                ))

        predictions = np.concatenate(predictions).reshape((-1, 1))
        y_labels = np.argmax(y, axis=1).reshape((-1, 1))
        prediction_labels = np.hstack([predictions, y_labels])
        with h5py.File(self.output()['predictions'].path, 'w') as hf:
            hf.create_dataset('data', data=prediction_labels)

        train_predictions = np.concatenate(train_predictions).reshape((-1, 1))
        train_y_labels = np.argmax(train_y, axis=1).reshape((-1, 1))

        metrics = {
            'accuracy': np.sum(np.equal(predictions, y_labels)) / len(y_labels),
            'train_accuracy': np.sum(np.equal(train_predictions, train_y_labels)) / len(train_y_labels)
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

    batch_size = luigi.IntParameter(default=128)
    num_epochs = luigi.IntParameter(default=100)


@luigi.util.requires(LogMelSpectrogramResNetCustom)
class ValidateLogMelSpectrogramResNetCustom(ValidateTensorflowModel):

    '''Validates LogMelSpectrogramResNet

    '''

    batch_size = luigi.IntParameter(default=128)
    num_epochs = luigi.IntParameter(default=100)


@luigi.util.requires(LogMelSpectrogramConvNet)
class ValidateLogMelSpectrogramConvNet(ValidateTensorflowModel):

    '''Validates LogMelSpectrogramResNet

    '''

    batch_size = luigi.IntParameter(default=128)
    num_epochs = luigi.IntParameter(default=40)


@luigi.util.requires(LogMelSpectrogramResNetv2)
class ValidateLogMelSpectrogramResNetv2(ValidateTensorflowModel):

    '''Validates LogMelSpectrogramResNetv2

    '''

    batch_size = luigi.IntParameter(default=128)
    num_epochs = luigi.IntParameter(default=40)


@luigi.util.inherits(data.DoDataPreProcessing)
class InitiateTraining(luigi.Task):

    '''Initiates training of all models

    This task act a gateway/middle-man between pre-processing and training.
    '''

    def requires(self):
        return {
            'clean': self.clone(data.DoDataPreProcessing),
            'noisy': self.clone(data.MixBackgroundWithRecordings),
        }


    def run(self):
        tasks = [
            TrainAllModels(
                data_files=[target.path for target in v['data']],
                label_files=[target.path for target in v['labels']]
            )
            for v in self.input().values()
        ]
        yield tasks


class TrainAllModels(luigi.Task):
    data_files = luigi.ListParameter()
    label_files = luigi.ListParameter()

    validation_tasks = []
    train_all_data_task = []
    # validation_tasks = [ValidateMfccSpectrogramCNN,
    #                     ValidateLogMelSpectrogramCNN]
    # train_all_data_task = [MfccSpectrogramCNN,
    #                        LogMelSpectrogramCNN,
    #                        LogMelSpectrogramResNet]
    # Some models are not configurable due to old implementation. In an
    # effort to not delete previously trained data, None is use to hack
    # older implementations into compatibility.
    model_settings = [
        None,
        None,
        {'block_sizes': [3, 3, 3],
         'filters': [16, 32, 64]}
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
            models.append(ValidateLogMelSpectrogramResNetCustom(
                data_files=data_subset[:1],
                label_files=labels_subset[:1],
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
                    validation_data=[],
                    validation_labels=[],
                ))
            else:
                models.append(task_class(
                    data_files=self.data_files,
                    label_files=self.label_files,
                    validation_data=[],
                    validation_labels=[],
                    model_settings=settings,
                ))
            models.append(ValidateLogMelSpectrogramResNetCustom(
                data_files=self.data_files,
                label_files=self.label_files,
                validation_data=[],
                validation_labels=[],
                model_settings=self.model_settings[2]
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


@luigi.util.inherits(data.DoDataPreProcessing)
class ExperimentBase(luigi.Task):

    def moel_tasks(self):
        raise NotImplementedError

    def requires(self):
        return {
            'clean': self.clone(data.DoDataPreProcessing),
            'noisy': self.clone(data.MixBackgroundWithRecordings),
            'noisy_0': self.clone(data.MixBackgroundWithRecordings,
                                  percentage=0.0),
            'noisy_0.1': self.clone(data.MixBackgroundWithRecordings,
                                    percentage=0.1),
            'noisy_0.8': self.clone(data.MixBackgroundWithRecordings,
                                    percentage=0.8),
        }

    def output(self):
        return [task.output() for task in self.model_tasks()]

    def run(self):
        yield self.model_tasks()


class Experiment1(ExperimentBase):

    '''Compare the difference between old hacked models and newer ResNet
    model

    Constants:
        Models: `LogMelSpectrogramResNet` OR
                `LogMelSpectrogramResNetConvNet`
        Learning rate: Piecewise, `0.1 -> 0.01 -> 0.001` @ epochs

    Variables:
        Data type: Clean or Noisy (`n_clean == n_noisy`)
        Spectrogram: Old configuration OR
                     Published configuration in:

        Tang, 2017. "Honk: A PyTorch Reimplementation of Convolution Neural
        Networks for Keyword Spotting."

    '''
    old_spectrogram_opts = {'frame_length': 128,
                            'frame_step': 64,
                            'fft_length': 1024,
                            'lower_hertz': 80.0,
                            'upper_hertz': 7600.0,
                            'num_mel_bins': 64}
    pub_spectrogram_opts = {'frame_step': 160,
                            'fft_length': 480,
                            'lower_hertz': 20.0,
                            'upper_hertz': 4000.0,
                            'num_mel_bins': 40}

    def model_tasks(self):
        resnet_config = list(itertools.product(
            ['clean', 'noisy'],
            [self.old_spectrogram_opts, self.pub_spectrogram_opts]
        ))

        hacked_resnet_tasks = [
            ValidateLogMelSpectrogramResNet(
                data_files=[t.path for t in
                            self.input()[data_type]['data'][:-1]],
                label_files=[t.path for t in
                             self.input()[data_type]['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()[data_type]['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()[data_type]['labels'][-1:]],
                model_settings={'spectrogram_opts': spectrogram_opts,
                                'resnet_size': 20},
            )
            for (data_type, spectrogram_opts) in resnet_config
        ]
        custom_resnet_tasks = [
            ValidateLogMelSpectrogramConvNet(
                data_files=[t.path for t in
                            self.input()[data_type]['data'][:-1]],
                label_files=[t.path for t in
                             self.input()[data_type]['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()[data_type]['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()[data_type]['labels'][-1:]],
                model_settings={'spectrogram_opts': spectrogram_opts,
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1]}
            )
            for (data_type, spectrogram_opts) in resnet_config
        ]
        return hacked_resnet_tasks + custom_resnet_tasks


class Experiment2(ExperimentBase):

    '''Evaluate the difference between validating with noisy data and clean
    data

    Constants:
        Models: `LogMelSpectrogramResNetConvNet`
        Epochs: 20, reduced to minimize runtime. Overfitting started around
            epoch 15-17 in Experiment1 for `LogMelSpectrogramConvNet`.
        Spectrogram: Published configuration in

            Tang, 2017. "Honk: A PyTorch Reimplementation of Convolution
            Neural Networks for Keyword Spotting."

    Variables:
        Validation Data type: Clean or Noisy (`n_clean == n_noisy`)

    '''
    spectrogram_opts = {'frame_step': 160,
                        'fft_length': 480,
                        'lower_hertz': 20.0,
                        'upper_hertz': 4000.0,
                        'num_mel_bins': 40}

    def model_tasks(self):
        convnet_tasks = [
            ValidateLogMelSpectrogramConvNet(
                data_files=[t.path for t in
                            self.input()['noisy']['data'][:-1]],
                label_files=[t.path for t in
                             self.input()['noisy']['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()[data_type]['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()[data_type]['labels'][-1:]],
                model_settings={'spectrogram_opts': self.spectrogram_opts,
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1]},
                num_epochs=20,
            )
            for data_type in ['clean', 'noisy']
        ]
        return convnet_tasks


class Experiment3(ExperimentBase):

    '''Tests convergence of ConvNet using clean validation data.

    Continuation of Experiment 2 to see if severe overfitting occurs at
    later epochs

    Constants:
        Models: `LogMelSpectrogramResNetConvNet`
        Spectrogram: Published configuration in

            Tang, 2017. "Honk: A PyTorch Reimplementation of Convolution
            Neural Networks for Keyword Spotting."

    Variables:
        Epochs: 20 vs 40. Evaluate if severe overfitting occurs

    '''
    spectrogram_opts = {'frame_step': 160,
                        'fft_length': 480,
                        'lower_hertz': 20.0,
                        'upper_hertz': 4000.0,
                        'num_mel_bins': 40}

    def model_tasks(self):
        convnet_tasks = [
            ValidateLogMelSpectrogramConvNet(
                data_files=[t.path for t in
                            self.input()['noisy']['data'][:-1]],
                label_files=[t.path for t in
                             self.input()['noisy']['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()['clean']['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()['clean']['labels'][-1:]],
                model_settings={'spectrogram_opts': self.spectrogram_opts,
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1]},
                num_epochs=num_epochs,
            )
            for num_epochs in [20, 40]
        ]
        return convnet_tasks


class Experiment4(ExperimentBase):

    '''Evaluate the effect of dropout on accuracy

    Dropout should slow down the rate of overfitting on training data and
    improve validation metrics.

    Constants:
        Models: `LogMelSpectrogramResNetConvNet`
        Spectrogram: Published configuration in

            Tang, 2017. "Honk: A PyTorch Reimplementation of Convolution
            Neural Networks for Keyword Spotting."
        Epochs: 50. Slightly longer to allow train accuracy to converge
            based on previous experiments.

    Variables:
        Dropout Rate: 0.20 (Comparing against previous experiments instead
        of running a new model with old parametes to save time)

    '''
    spectrogram_opts = {'frame_step': 160,
                        'fft_length': 480,
                        'lower_hertz': 20.0,
                        'upper_hertz': 4000.0,
                        'num_mel_bins': 40}

    def model_tasks(self):
        convnet_tasks = [
            ValidateLogMelSpectrogramConvNet(
                data_files=[t.path for t in
                            self.input()['noisy']['data'][:-1]],
                label_files=[t.path for t in
                             self.input()['noisy']['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()['clean']['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()['clean']['labels'][-1:]],
                model_settings={'spectrogram_opts': self.spectrogram_opts,
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1]},
                num_epochs=50,
                dropout_rate=dropout_rate
            )
            for dropout_rate in [0.20]
        ]
        return convnet_tasks


class Experiment5(ExperimentBase):

    '''Evaluate a second attempt at ResNet

    Checking the hypothesis that a two layer ConvNet has trouble storing
    enough information. Hoping a ResNet does not have this trouble.

    Constants:
        Models: `LogMelSpectrogramResNetv2`
        Spectrogram: Published configuration in

            Tang, 2017. "Honk: A PyTorch Reimplementation of Convolution
            Neural Networks for Keyword Spotting."
        Epochs: 50. Slightly longer to allow train accuracy to converge
            based on previous experiments.
        Dropout Rate: 0.20 (Comparing against previous experiments instead
            of running a new model with old parametes to save time)

    '''
    spectrogram_opts = {'frame_step': 160,
                        'fft_length': 480,
                        'lower_hertz': 20.0,
                        'upper_hertz': 4000.0,
                        'num_mel_bins': 40}

    def model_tasks(self):
        convnet_tasks = [
            ValidateLogMelSpectrogramResNetv2(
                data_files=[t.path for t in
                            self.input()['noisy']['data'][:-1]],
                label_files=[t.path for t in
                             self.input()['noisy']['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()['clean']['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()['clean']['labels'][-1:]],
                model_settings={'spectrogram_opts': self.spectrogram_opts,
                                'block_sizes': [3, 3],
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1]},
                num_epochs=50,
                dropout_rate=dropout_rate
            )
            for dropout_rate in [0.20]
        ]
        return convnet_tasks


class Experiment6(ExperimentBase):

    '''Re-evaluate the effect of dropout on accuracy at a lower learning
    rate

    Loss curve suggest LR is too high (see
    http://cs231n.github.io/neural-networks-3/).

    Constants:
        Models: `LogMelSpectrogramResNetConvNet`
        Spectrogram: Published configuration in

            Tang, 2017. "Honk: A PyTorch Reimplementation of Convolution
            Neural Networks for Keyword Spotting."
        Epochs: 50. Slightly longer to allow train accuracy to converge
            based on previous experiments.
        Dropout Rate: 0.20 (Comparing against previous experiments instead
        of running a new model with old parametes to save time)

    Variables:
        Learning rate: 1e-6 (vs original 0.005)

    '''
    spectrogram_opts = {'frame_step': 160,
                        'fft_length': 480,
                        'lower_hertz': 20.0,
                        'upper_hertz': 4000.0,
                        'num_mel_bins': 40}

    def model_tasks(self):
        convnet_tasks = [
            ValidateLogMelSpectrogramConvNet(
                data_files=[t.path for t in
                            self.input()['noisy']['data'][:-1]],
                label_files=[t.path for t in
                             self.input()['noisy']['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()['clean']['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()['clean']['labels'][-1:]],
                model_settings={'spectrogram_opts': self.spectrogram_opts,
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1],
                                'initial_learning_rate': 1e-6},
                num_epochs=50,
                dropout_rate=dropout_rate
            )
            for dropout_rate in [0.20]
        ]
        return convnet_tasks


class Experiment7(ExperimentBase):

    '''Evaluate the effect of noisy (but less noisy) training data

    Constants:
        Models: `LogMelSpectrogramResNetConvNet`
        Spectrogram: Published configuration in

            Tang, 2017. "Honk: A PyTorch Reimplementation of Convolution
            Neural Networks for Keyword Spotting."
        Epochs: 50. Slightly longer to allow train accuracy to converge
            based on previous experiments.
        Dropout Rate: 0.20 (Comparing against previous experiments instead
        of running a new model with old parametes to save time)

    Variables:
        Training data: 80% of data have random background noise added.

    '''
    spectrogram_opts = {'frame_step': 160,
                        'fft_length': 480,
                        'lower_hertz': 20.0,
                        'upper_hertz': 4000.0,
                        'num_mel_bins': 40}

    def model_tasks(self):
        convnet_tasks = [
            ValidateLogMelSpectrogramConvNet(
                data_files=[t.path for t in
                            self.input()['noisy_0.1']['data'][:-1]],
                label_files=[t.path for t in
                             self.input()['noisy_0.1']['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()['clean']['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()['clean']['labels'][-1:]],
                model_settings={'spectrogram_opts': self.spectrogram_opts,
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1],
                                'initial_learning_rate': 0.01},
                num_epochs=40,
                dropout_rate=0.2
            )
        ]
        return convnet_tasks
