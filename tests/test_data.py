'''Tests tfspeech.tasks.data

'''
import shutil
import tempfile

import numpy as np
import pytest

import tfspeech.tasks.data


@pytest.fixture
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


def test_CreateSilenceSnippets(temp_dir):
    task = tfspeech.tasks.data.CreateSilenceSnippets(
        num_samples=2,
        base_dir=temp_dir,
    )

    task.run()

    # TODO: Need to check the target contents
    # TODO: Test expect data files from Kaggle already be downloaded
    assert task.complete()


def test_MixBackgroundWithRecordings__add_noise_and_write_no_noise():
    task = tfspeech.tasks.data.MixBackgroundWithRecordings(
        num_partitions=2,
        data_directories=[],
        percentage=0.0
        )

    data = np.random.rand(5, 100)
    labels = np.random.randint(0, 2, size=(5, ))

    backgrounds = [np.ones((1000, )) * 0.5,
                   np.ones((1000, )) * 0.5]

    noised_data, noised_labels = task._add_noise(
        data, labels,
        backgrounds, [0.1, 0.1 + 1e-9])

    expected = data

    assert np.allclose(noised_data, expected)
    assert np.allclose(noised_labels, labels)


def test_MixBackgroundWithRecordings__add_noise_and_write():
    task = tfspeech.tasks.data.MixBackgroundWithRecordings(
        num_partitions=2,
        data_directories=[],
        )

    data = np.random.rand(5, 100)
    labels = np.random.randint(0, 2, size=(5, ))

    backgrounds = [np.ones((1000, )) * 0.5,
                   np.ones((1000, )) * 0.5]

    noised_data, noised_labels = task._add_noise(
        data, labels,
        backgrounds, [0.1, 0.1 + 1e-9])

    expected = data + 0.05

    assert np.allclose(noised_data, expected)
    assert np.allclose(noised_labels, labels)
