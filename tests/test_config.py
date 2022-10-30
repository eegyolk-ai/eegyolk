# -*- coding: utf-8 -*-

"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains one method to let the user configure
all paths instead of hard-coding them.
"""

import json
import os

from contextlib import contextmanager
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from eegyolk.config import Config


class TestConfig(TestCase):

    required_directories = {
        'root': ('data', 'metadata'),
        'root_2022': ('data', 'metadata'),
    }

    @contextmanager
    def cd(self, directory=None):
        if not directory:
            with TemporaryDirectory() as td:
                old = os.getcwd()
                os.chdir(td)
                yield td
                os.chdir(old)
        else:
            old = os.getcwd()
            os.chdir(td)
            yield td
            os.chdir(old)

    def test_roots_only(self):
        with TemporaryDirectory() as td:
            os.mkdir(os.path.join(td, 'root'))
            os.mkdir(os.path.join(td, 'root_2022'))
            raw_config = {
                'root': os.path.join(td, 'root'),
                'root_2022': os.path.join(td, 'root_2022'),
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)

            for root, dirs in self.required_directories.items():
                for d in dirs:
                    os.mkdir(os.path.join(td, root, d))

            config = Config(config_file)
            assert config.get_directory('data_2022')

    def test_data_only(self):
        with TemporaryDirectory() as td:
            raw_config = {
                'data': 'data/',
                'preprocessed': 'preprocessed/',
                'metadata': 'metadata/',
                'models': 'models/',
                'data_2022': 'data-2022/',
                'preprocessed_2022': 'preprocessed-2022/',
                'metadata_2022': 'metadata-2022/',
                'models_2022': 'models-2022/',
            }
            raw_config = {
                k: os.path.join(td, v) for k, v in raw_config.items()
            }
            for v in raw_config.values():
                os.mkdir(os.path.join(td, v))
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)
            config = Config(config_file)
            assert config.get_directory('data_2022')

    def test_override_some(self):
        with TemporaryDirectory() as td:
            raw_config = {
                'root': 'root',
                'data': 'data/',
                'root_2022': 'root_2022',
                'data_2022': 'data-2022/',
            }
            raw_config = {
                k: os.path.join(td, v) for k, v in raw_config.items()
            }
            for v in raw_config.values():
                os.mkdir(os.path.join(td, v))
            os.mkdir(os.path.join(td, 'root', 'metadata'))
            os.mkdir(os.path.join(td, 'root_2022', 'metadata'))
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)
            config = Config(config_file)
            assert config.get_directory('metadata_2022')

    def test_missing_config_path(self):
        try:
            Config('non existent')
        except Exception as e:
            assert 'Cannot find config files' in e.args[0]
        else:
            assert False, 'Didn\'t notify on missing config file'

    def test_default_config_path(self):
        with self.cd() as td:
            with open(os.path.join(td, 'config.json'), 'w') as f:
                f.write('x')
            try:
                Config()
            except Exception as e:
                assert 'Found invalid JSON in ./config.json' == e.args[0]
            else:
                assert False, 'Didn\'t notify on missing config file'

    def test_incorrect_data_path(self):
        with TemporaryDirectory() as td:
            os.mkdir(os.path.join(td, 'root'))
            os.mkdir(os.path.join(td, 'root_2022'))
            raw_config = {
                'root': 'root',
                'root_2022': 'root_2022',
            }
            raw_config = {
                k: os.path.join(td, v) for k, v in raw_config.items()
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)
            try:
                config = Config(config_file)
            except ValueError as e:
                expected_missing = []
                for root, dirs in self.required_directories.items():
                    for d in dirs:
                        full_path = os.path.join(raw_config[root], d)
                        if full_path not in e.args[0]:
                            expected_missing.append(d)
                assert not expected_missing, (
                    'Excpected to miss: {}'.format(missing)
                )
            else:
                assert False, 'Failed to identify missing data directory'


if __name__ == '__main__':
    main()
