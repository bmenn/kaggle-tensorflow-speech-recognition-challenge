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
            '_'.join([background, str(i)]) + '.wav'
            for i, background
            in zip(range(self.num_samples),
                   itertools.cycle(self.input().keys()))
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
            background_label = target.path.split('/')[-1].split('_')[0]
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
