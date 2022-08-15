# -*- coding: utf-8 -*-

"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains one method to let the user configure
all paths instead of hard-coding them.
"""

import json
import logging
import os
import textwrap


class Config:

    default_locations = (
        './config.json',
        os.path.expanduser('~/.epodium/config.json'),
        '/etc/epodium/config.json',
    )

    default_layout = {
        'root': '{}',
        'data': '{}/data',
        'metadata': '{}/metadata',
        'preprocessed': '{}/preprocessed',
        'models': '{}/models',
    }

    required_directories = 'data', 'metadata'

    def __init__(self, location=None):
        self._raw = None
        self._loaded = None
        self.load(location)
        self.validate()

    def usage(self):
        return textwrap.dedent(
            '''
            Cannot load config.

            Please create a file in either one of the locations
            listed below:
            {}

            With the contents that specifies at least the root
            directory as follows:

            {{
                "root": "/path/to/storage"
            }}

            The default directory layout is expected to be:

            {{
                "root": "/path/to/storage",
                "data": "$root/data/",
                "preprocessed": "$root/preprocessed/",
                "metadata": "$root/metadata/",
                "models": "$root/models/"
            }}

            You can override any individual directory by specifying it
            in config.json.

            "data" and "metadata" directories are expected to exist.
            The "data" directory is expected to have "11mnd mmn" ..
            "47mnd mmn" directories.  The "metadata" directory is
            expected to have "ages" subdirectory with files named
            "ages_11mnths.txt" .. "ages_47mnths.txt".

            The "models" and "preprocessed" directories need not
            exist.  They will be created if missing.  The "preprocessed"
            directory will be used to output CSV and h5 files when
            reading the raw CNT data.  The "models" directory will
            be used to store regression and neural network models
            created by training the models on preprocessed data.
            '''
        ).format('\n'.join(self.default_locations))

    def load(self, location):
        if location is not None:
            with open(location) as f:
                self._raw = json.load(f)
                return

        for p in self.default_locations:
            try:
                with open(p) as f:
                    self._raw = json.load(f)
                    break
            except Exception as e:
                logging.info('Failed to load %s: %s', p, e)
        else:
            raise ValueError(self.usage())

        root = self._raw.get('root')
        self._loaded = dict(self._raw)
        if root is None:
            required = dict(self.default_layout)
            del required['root']
            for directory in required.keys():
                if directory not in self._raw:
                    raise ValueError(self.usage())
            # User specified all concrete directories.  Nothing for us to
            # do here.
        else:
            missing = set(self.default_layout.keys()) - set(self._raw.keys())
            # User possibly specified only a subset of directories.  We'll
            # back-fill all the not-specified directories.
            for m in missing:
                self._loaded[m] = self.default_layout[m].format(root)

    def validate(self):
        for d in self.required_directories:
            if not os.path.isdir(self._loaded[d]):
                logging.error('Directory %s must exist', self._loaded[d])
                raise ValueError(self.usage())

    def get_directory(self, directory, value=None):
        if value is None:
            return self._loaded[directory]
        return value
