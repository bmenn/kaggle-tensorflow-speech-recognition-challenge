'''Tests tfspeech.tasks.data

'''
import shutils
import tempfile

import tfspeech.tasks.data


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
    assert task.complete()
