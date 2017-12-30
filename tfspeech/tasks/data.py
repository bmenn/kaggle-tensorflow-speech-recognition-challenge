'''Create/fetch data for training

'''
import itertools
import random

import luigi
import scipy.io.wavfile


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
