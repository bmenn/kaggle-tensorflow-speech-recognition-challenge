'''Utility functions that do not have home elsewhere

'''
import hashlib
import os

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops


def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  # Stolen from: tensorflow example `speed_commands` input_data.py
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def create_validation_groups(filenames, num_partitions):
    '''Splits a list of filenames into partitions for cross-validation

    Expected paths in the following structure
    `foo/bar/SPEECH_LABEL/SPEAKER_foo_ID.wav`.

    `SPEAKER_foo_ID.wav` actually expects the portions of the filename to be
    separated by `_`. Files are partitioned so that no speaker is split up
    over multiple partitions to prevent overfitting. As a result, partitions
    may not be equal in size.

    :filenames: List of filenames to split into groups
    :returns: List of lists containing files

    '''
    partitions = [[] for i in range(num_partitions)]
    for path in filenames:
        _, fname = os.path.split(path)

        speaker = fname.split('_')[0]
        speaker_hash = hashlib.sha1(speaker.encode('utf-8')).hexdigest()
        partition_id = int(speaker_hash, 16) % num_partitions

        partitions[partition_id].append(path)

    return partitions
