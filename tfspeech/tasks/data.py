'''Create/fetch data for training

'''
import glob
import itertools
import os
import random

import h5py
import luigi
import luigi.util
import numpy as np
import scipy.io.wavfile

import tfspeech.models as models
import tfspeech.utils as utils


LABELS = models.LABELS


class BackgroundNoiseRecordings(luigi.ExternalTask):

    '''Placeholder task for background recording tasks

    '''

    def output(self):
        return {
            'dishes': luigi.LocalTarget('data/train/audio/_background_noise_/doing_the_dishes.wav'),
            'dude-miaowing': luigi.LocalTarget('data/train/audio/_background_noise_/dude_miaowing.wav'),
            'exercise-bike': luigi.LocalTarget('data/train/audio/_background_noise_/exercise_bike.wav'),
            'pink-noise': luigi.LocalTarget('data/train/audio/_background_noise_/pink_noise.wav'),
            'running-tap': luigi.LocalTarget('data/train/audio/_background_noise_/running_tap.wav'),
            'white-noise': luigi.LocalTarget('data/train/audio/_background_noise_/white_noise.wav'),
        }


class CreateSilenceSnippets(luigi.Task):

    '''Create silence data samples for training

    '''

    num_samples = luigi.IntParameter()
    sample_length = luigi.FloatParameter(default=1.0)
    base_dir = luigi.Parameter(default='data/samples')

    def requires(self):
        # TODO: Need ability to inject alternative external task for testing
        return BackgroundNoiseRecordings()

    def output(self):
        filenames = [
            '_'.join([str(i),
                      background,
                      str(i)]) + '.wav'
            for i, background
            in zip(range(self.num_samples),
                   itertools.cycle(sorted(self.input().keys())))
        ]
        return [luigi.LocalTarget('/'.join([self.base_dir, 'silence', fname]))
                for fname in filenames]

    def run(self):
        recordings = {}
        for background_label, target in self.input().items():
            sample_rate, data = scipy.io.wavfile.read(target.path)
            recordings[background_label] = {
                'sample_rate': sample_rate,
                'data': data,
            }


        for target in self.output():
            background_label = target.path.split('/')[-1].split('_')[1]
            recording = recordings[background_label]
            sample_rate = recording['sample_rate']
            data = recording['data']
            length = int(sample_rate * self.sample_length)

            start = random.randint(0, len(data) - length)
            target.makedirs()
            scipy.io.wavfile.write(target.path, sample_rate,
                                   data[start:start+length])


class DataPartition(luigi.ExternalTask):

    '''Docstring for DataPartition. '''

    base_dir = luigi.Parameter(default='data')
    partition_id = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.base_dir,
                'partitions/{id}/files.txt'.format(id=self.partition_id),  # pylint: disable=no-member
            )
        )


@luigi.util.requires(DataPartition)
class ConvertWavToArray(luigi.Task):

    '''Convert wavfiles to arrays of floats'''

    base_dir = luigi.Parameter(default='data')
    resources = {'tensorflow': 1}

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.base_dir,
                'partitions/{id}/data.h5'.format(id=self.partition_id),  # pylint: disable=no-member
            )
        )

    def run(self):
        with self.input().open('r') as f:
            filenames = [l.strip() for l in f.readlines()]

        data = utils.load_wav_file(filenames)

        with h5py.File(self.output().path, 'w') as hf:
            hf.create_dataset('data', data=data)


@luigi.util.requires(DataPartition)
class ConvertFilenameToLabel(luigi.Task):

    '''Encode filename labels to one-hot arrays'''

    base_dir = luigi.Parameter(default='data')

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.base_dir,
                'partitions/{id}/label.h5'.format(id=self.partition_id),  # pylint: disable=no-member
            )
        )

    def run(self):
        with self.input().open('r') as f:
            filenames = [l.strip() for l in f.readlines()]

        labels = []
        lookup = dict((l, i) for i, l in enumerate(LABELS))
        for path in filenames:
            label = utils.parse_path_metadata(path)['label']
            try:
                labels.append(lookup[label])
            except KeyError:
                labels.append(lookup['unknown'])

        data = np.zeros((len(labels), len(LABELS)))
        data[np.arange(len(labels)), labels] = 1
        with h5py.File(self.output().path, 'w') as hf:
            hf.create_dataset('data', data=data)


class PartitionDataFiles(luigi.Task):

    '''Partitions data files'''

    num_partitions = luigi.IntParameter()
    data_directories = luigi.ListParameter()
    base_dir = luigi.Parameter(default='data')

    def output(self):
        return [
            luigi.LocalTarget(
                os.path.join(
                    self.base_dir,
                    'partitions/{id}/files.txt'.format(id=i),  # pylint: disable=no-member
                )
            )
            for i in range(self.num_partitions)
        ]

    def run(self):
        filenames = []
        for data_directory in self.data_directories:
            search_dir = os.path.join(
                data_directory,
                '**/*.wav',
            )
            filenames.extend([
                os.path.abspath(path)
                for path in glob.iglob(search_dir)
                if '_background_noise_' not in path
            ])
        partitions = utils.create_validation_groups(filenames,
                                                    self.num_partitions)

        for target, paths in zip(self.output(), partitions):
            with target.open('w') as f:
                f.write('\n'.join(paths))


class DoDataPreProcessing(luigi.Task):

    '''Wrapper task to execute all the data pre-processing'''

    num_partitions = luigi.IntParameter()
    data_directories = luigi.ListParameter()
    base_dir = luigi.Parameter(default='data')

    def output(self):
        return {
            'data': [
                self.clone(ConvertWavToArray, partition_id=i).output()
                for i in range(self.num_partitions)
            ],
            'labels': [
                self.clone(ConvertFilenameToLabel, partition_id=i).output()
                for i in range(self.num_partitions)
            ],
        }

    def run(self):
        yield CreateSilenceSnippets(
            num_samples=2000,
        )

        yield self.clone(PartitionDataFiles)
        yield [
            self.clone(ConvertWavToArray, partition_id=i)
            for i in range(self.num_partitions)
        ] + [
            self.clone(ConvertFilenameToLabel, partition_id=i)
            for i in range(self.num_partitions)
        ]


@luigi.util.inherits(DoDataPreProcessing)
class MixBackgroundWithRecordings(luigi.Task):

    '''Wrapper task to execute all the data pre-processing'''

    num_partitions = luigi.IntParameter()
    data_directories = luigi.ListParameter()
    base_dir = luigi.Parameter(default='data')
    duplicate_count = luigi.IntParameter(default=1)
    percentage = luigi.FloatParameter(default=1.0)

    def requires(self):
        return {
            'partitions': self.clone(DoDataPreProcessing),
            'background': BackgroundNoiseRecordings(),
        }

    def output(self):
        return {
            'data': [
                luigi.LocalTarget(
                    os.path.split(t.path)[0]
                    + '/data_noised_%dx_%0.2f.h5' % (self.duplicate_count,
                                                     self.percentage)
                )
                for t in self.input()['partitions']['data']
            ],
            'labels': [
                luigi.LocalTarget(
                    os.path.split(t.path)[0]
                    + '/label_noised_%dx_%0.2f.h5' % (self.duplicate_count,
                                                      self.percentage)
                )
                for t in self.input()['partitions']['labels']
            ],
        }

    def _add_noise_and_write(self, data_path, label_path, backgrounds,
                             index, noise_multiplier=None):
        if noise_multiplier is None:
            noise_multiplier = [0.05, 0.2]

        with h5py.File(data_path, 'r') as hf:
            data = hf['data'][:]
        with h5py.File(label_path, 'r') as hf:
            labels = hf['data'][:]

        noised_data, noised_labels = self._add_noise(
            data, labels,
            backgrounds,
            noise_multiplier,
        )
        self._write_noise_data(noised_data, labels, index)

    def _add_noise(self, data, labels, backgrounds, noise_multiplier):
        noised_data = np.repeat(data, self.duplicate_count, axis=0)
        noised_labels = np.repeat(labels, self.duplicate_count, axis=0)

        sample_length = noised_data.shape[1]
        background_samples = np.zeros(noised_data.shape)
        num_backgrounds = len(backgrounds)
        for i in range(background_samples.shape[0]):
            selected_background = backgrounds[
                random.randrange(num_backgrounds - 1)]
            start = random.randrange(len(selected_background)
                                     - sample_length)
            if random.random() < self.percentage:
                try:
                    background_samples[i, :] = (
                        selected_background[start:start + sample_length]
                        * noise_multiplier
                    )
                except ValueError:
                    background_samples[i, :] = (
                        selected_background[start:start + sample_length]
                        * np.random.uniform(low=noise_multiplier[0],
                                            high=noise_multiplier[1],
                                            size=(1, ))
                    )
        return noised_data + background_samples, noised_labels

    def _write_noise_data(self, noised_data, labels, index):
        # Writing out here to abuse the stack and keep memory usage lower
        with h5py.File(self.output()['data'][index].path,
                       'w') as hf:
            hf.create_dataset('data', data=noised_data)
        with h5py.File(self.output()['labels'][index].path,
                       'w') as hf:
            hf.create_dataset('data', data=labels)

    def run(self):
        backgrounds = [scipy.io.wavfile.read(t.path)[1]
                       for t in self.input()['background'].values()]
        for i in range(self.num_partitions):
            data_target = self.input()['partitions']['data'][i]
            label_target = self.input()['partitions']['labels'][i]

            self._add_noise_and_write(data_target.path, label_target.path,
                                      backgrounds, i)


class ConvertTestWavToArray(luigi.Task):

    '''Convert wavfiles to arrays of floats'''

    base_dir = luigi.Parameter(default='data')
    resources = {'tensorflow': 1}

    def output(self):
        return {
            'data': luigi.LocalTarget(
                os.path.join(
                    self.base_dir,
                    'test.h5',  # pylint: disable=no-member
                )
            ),
            'files': luigi.LocalTarget(
                os.path.join(
                    self.base_dir,
                    'test_files.txt',  # pylint: disable=no-member
                )
            )
        }

    def run(self):
        # TODO need to make this not dependent on current working directory
        filenames = sorted([
            os.path.abspath(path)
            for path in glob.iglob('data/test/audio/*.wav')
        ])

        # Batching otherwise memory becomes an issue
        data = []
        for i in range(0, len(filenames), 10000):
            data.append(utils.load_wav_file(filenames[i:i+10000]))
        data = np.vstack(data)

        with h5py.File(self.output()['data'].path, 'w') as hf:
            hf.create_dataset('data', data=data)
        with self.output()['files'].open('w') as f:
            f.write(
                '\n'.join([os.path.split(fname)[1] for fname in filenames])
            )
