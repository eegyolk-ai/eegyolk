#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


if __name__ == '__main__':
    setup(
        name='epodium-original',
        packages=(
            'epodium',
        ),
        install_requires=(
            'notebook==6.4.12',
            'nbconvert==6.4.4',
            'numpy',
            'pandas',
            'h5py',
            'matplotlib',
            'mne',
            'mne-features',
            'seaborn',
            # pytables for conda
            'tables',
            'sklearn-rvm',
            'tensorflow',
        ),
        setup_requires=(
            'wheel',
        ),
    )
