'''Tests tfspeech.tasks.data

'''
import shutil
import tempfile

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
