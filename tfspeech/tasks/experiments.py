"""Various experiments

"""
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
import tfspeech.tasks.train as train

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_sh = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)-15s %(message)s')
_sh.setFormatter(_formatter)
LOGGER.addHandler(_sh)


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
            train.ValidateLogMelSpectrogramResNet(
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
            train.ValidateLogMelSpectrogramConvNet(
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
            train.ValidateLogMelSpectrogramConvNet(
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
            train.ValidateLogMelSpectrogramConvNet(
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
            train.ValidateLogMelSpectrogramConvNet(
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
            train.ValidateLogMelSpectrogramResNetv2(
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
            train.ValidateLogMelSpectrogramConvNet(
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
            train.ValidateLogMelSpectrogramConvNet(
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


class Experiment8(ExperimentBase):

    '''Evaluate the effect of dynamically augmenting training data

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
        Percent of augmented data: 80% (with changing augmentation) vs
            statically augmented data (if augmentation is applied)
        Noise volume: 10% of random background noise is added to training
            samples selected for augmentation.

    '''
    spectrogram_opts = {'frame_step': 160,
                        'fft_length': 480,
                        'lower_hertz': 20.0,
                        'upper_hertz': 4000.0,
                        'num_mel_bins': 40}

    def model_tasks(self):
        convnet_tasks = [
            train.ValidateLogMelSpectrogramConvNet(
                data_files=[t.path for t in
                            self.input()['clean']['data'][:-1]],
                label_files=[t.path for t in
                             self.input()['clean']['labels'][:-1]],
                validation_data=[t.path for t in
                                 self.input()['clean']['data'][-1:]],
                validation_labels=[t.path for t in
                                   self.input()['clean']['labels'][-1:]],
                model_settings={'spectrogram_opts': self.spectrogram_opts,
                                'filters': [64, 64],
                                'kernel_sizes': [[20, 8], [10, 4]],
                                'max_pool_sizes': [2, 1],
                                'pool_strides': [2, 1],
                                'initial_learning_rate': 0.001},
                num_epochs=40,
                dropout_rate=0.2,
                percentage=0.8,
                noise_volume=0.1,
            )
        ]
        return convnet_tasks
