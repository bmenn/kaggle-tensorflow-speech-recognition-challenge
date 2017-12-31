from setuptools import setup, find_packages


INSTALL_REQUIREMENTS = [
    'tensorflow-gpu',
    'luigi',
    'scipy',
    'h5py',
]
TESTS_REQUIREMENTS = [
    'pytest-runner',
    'pytest',
    'pytest-testmon',
    'pytest-watch',
    'pylint',
]

setup(
    name='tf-speech',
    packages=find_packages(),
    install_requires=INSTALL_REQUIREMENTS,
    tests_require=TESTS_REQUIREMENTS,
    extras_require={
        'all': INSTALL_REQUIREMENTS,
        'test': TESTS_REQUIREMENTS,
    },
)
