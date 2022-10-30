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

from json import JSONDecodeError


class Config:

    default_locations = (
        './config.json',
        os.path.expanduser('~/.eegyolk/config.json'),
        '/etc/eegyolk/config.json',
    )

    default_layout = {
        'root': '{}',
        'data': '{}/data',
        'metadata': '{}/metadata',
        'preprocessed': '{}/preprocessed',
        'models': '{}/models',
        'root_2022': '{}',
        'data_2022': '{}/data',
        'metadata_2022': '{}/metadata',
        'preprocessed_2022': '{}/preprocessed',
        'models_2022': '{}/models',
    }

    required_directories = 'data', 'metadata', 'data_2022', 'metadata_2022'

    roots = {
        'root': ('data', 'metadata', 'preprocessed', 'models'),
        'root_2022': (
            'data_2022',
            'metadata_2022',
            'preprocessed_2022',
            'models_2022',
        ),
    }

    def __init__(self, location=None):
        self._raw = None
        self._loaded = None
        self.load(location)

    def usage(self, message):
        return textwrap.dedent(
            '''
            Cannot load config: {}

            Please create a file in either one of the locations
            listed below:
            {}

            With the contents that specifies at least the root
            directories as follows:

            {{
                "root": "/path/to/storage",
                "root_2022": "/path/to/storage"
            }}

            The default directory layout is expected to be:

            {{
                "root": "/path/to/storage",
                "data": "$root/data/",
                "preprocessed": "$root/preprocessed/",
                "metadata": "$root/metadata/",
                "models": "$root/models/",
                "root_2022": "/path/to/storage-2022",
                "data_2022": "$root_2022/data/",
                "preprocessed_2022": "$root_2022/preprocessed/",
                "metadata_2022": "$root_2022/metadata/",
                "models_2022": "$root_2022/models/"
            }}

            You can override any individual directory by specifying it
            in config.json.

            "data" and "metadata" directories are expected to exist.
            The "data" directory is expected to have "11mnd mmn" ..
            "47mnd mmn" directories.  The "metadata" directory is
            expected to have "ages" subdirectory with files named
            "ages_11mnths.txt" .. "ages_47mnths.txt".

            The "models" and "preprocessed" directories and their 2022
            counterparts need not exist.  They will be created if
            missing.  The "preprocessed" directory will be used to
            output CSV and h5 files when reading the raw CNT data.
            The "models" directory will be used to store regression
            and neural network models created by training the models
            on preprocessed data.'''
        ).format(message, '\n'.join(self.default_locations))

    def load(self, location):
        locations = [location] if location else self.default_locations

        for p in locations:
            try:
                with open(p) as f:
                    raw = json.load(f)
                    break
            except JSONDecodeError as e:
                raise ValueError(
                    'Found invalid JSON in {}'.format(p)
                ) from e
            except Exception as e:
                logging.info('Failed to load %s: %s', p, e)
        else:
            raise ValueError(self.usage('Cannot find config files'))

        merged = self.merge(raw)
        self.validate(merged)
        self._raw = raw
        self._loaded = merged

    def merge(self, raw):
        result = {}
        for root, derived in self.roots.items():
            root_path = raw.get(root)
            if root_path is not None:
                result[root] = root_path
                for d in derived:
                    result[d] = self.default_layout[d].format(root_path)
        result.update(raw)
        return result

    def validate(self, merged):
        missing = []
        for d in self.required_directories:
            required = merged[d]
            if not os.path.isdir(required):
                logging.error('Directory %s must exist', required)
                missing.append(required)

        if missing:
            raise ValueError(
                self.usage('Missing: {}'.format(', '.join(missing))),
            )

    def get_directory(self, directory, value=None):
        if value is None:
            return self._loaded[directory]
        return value
