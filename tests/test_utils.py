'''Tests tfspeech.util module

'''
import tfspeech.utils


def test_create_validation_groups():
    '''`tfspeech.utils.create_validation_groups` SHALL partition filenames
    such that no speaker is split across multiple labels.

    '''
    filenames = [
        'data/yes/A_nohash_1.wav',
        'data/no/A_nohash_1.wav',
        'data/no/A_nohash_2.wav',
        'data/yes/B_nohash_1.wav',
        'data/no/B_nohash_1.wav',
    ]
    num_partitions = 2

    partitions = tfspeech.utils.create_validation_groups(filenames,
                                                         num_partitions)
    expected = [
        [
            'data/yes/A_nohash_1.wav',
            'data/no/A_nohash_1.wav',
            'data/no/A_nohash_2.wav',
        ],
        [
            'data/yes/B_nohash_1.wav',
            'data/no/B_nohash_1.wav',
        ],
    ]

    assert (sorted([sorted(p) for p in partitions], key=lambda p: p[0]) ==
            sorted([sorted(p) for p in expected], key=lambda p: p[0]))


def test_parse_path_metadata():
    '''`tfspeech.utils.parse_path_metadata` SHALL retrieve speaker
    (`SPEAKER`), label (`LABEL`), and instance count (`COUNT`) from path in
    the following format:

        foo/bar/LABEL/SPEAKER_baz_COUNT.wav

    '''
    path = 'foo/bar/LABEL/SPEAKER_baz_1.wav'

    metadata = tfspeech.utils.parse_path_metadata(path)
    expected = {'speaker': 'SPEAKER', 'count': 1, 'label': 'LABEL'}

    assert metadata == expected
