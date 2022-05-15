# testing eegyolk

import unittest
import os
import glob
import sys
from tempfile import TemporaryDirectory
import mne

sys.path.insert(0, os.getcwd())

from eegyolk.display_helper import make_ordinal
from eegyolk.helper_functions import band_pass_filter
from eegyolk.helper_functions import hash_it_up_right_all
from eegyolk.helper_functions import load_metadata # (filename, path_metadata, path_output, make_excel_files=True, make_csv_files=True)

from eegyolk.initialization_functions import load_dataset
# from eegyolk.initialization_functions import load_metadata
from eegyolk.initialization_functions import load_event_markers
from eegyolk.initialization_functions import save_event_markers
from eegyolk.initialization_functions import print_event_info

sample_eeg = 'tests/sample/640-464-17m-jc-mmn36.cnt'
sample_eeg_read = mne.io.read_raw_cnt(sample_eeg, preload=True)
sample_metadata = 'tests/sample/cdi.txt'
event_marker_folder = 'tests/sample/fake_event_markers'
sample_eeg_list = ['101a','101b']

class TestDisplayHelperMethods(unittest.TestCase):

    def test_make_ordinal(self):
        self.assertEqual(make_ordinal(5), '5th')


class TestFilteringMethods(unittest.TestCase):

    def test_band_pass_filter(self):
        sample_eeg_filtered = band_pass_filter(sample_eeg_read, 0, 10)
        self.assertEqual(
            (sample_eeg_filtered.info['lowpass']),
            10,
        )


class TestHashMethods(unittest.TestCase):

    def test_hash_it_up_right_all(self):
        tempfile1 = 'tempfile1.cnt'
        tempfile2 = 'tempfile2.cnt'
        with TemporaryDirectory() as td:
            with open(os.path.join(td, tempfile1), 'w') as tf:
                tf.write('string')
            with open(os.path.join(td, tempfile2), 'w') as tf:
                tf.write('string')
            self.assertTrue(hash_it_up_right_all(td, '.cnt').equals(hash_it_up_right_all(td, '.cnt')))

class TestLoadMethods(unittest.TestCase):

   
    def test_load_metadata(self):
        filename = os.path.splitext(sample_metadata)[0]
        loaded_metadata = load_metadata(
            filename,
            sys.path[0],
            sys.path[0],
            make_excel_files=False,
            make_csv_files=False,
        )
        self.assertEqual(len(loaded_metadata), 143)

    def test_load_event_markers(self):
        loaded_event_markers = load_event_markers(event_marker_folder, sample_eeg_list)
        self.assertEqual(len(loaded_event_markers), 2)

if __name__ == '__main__':
    unittest.main()

