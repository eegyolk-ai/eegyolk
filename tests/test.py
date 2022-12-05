# testing eegyolk

from posixpath import splitext
from itertools import islice
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
from eegyolk.helper_functions import filter_eeg_raw
from eegyolk.helper_functions import load_metadata 

from eegyolk.initialization_functions import load_dataset
# from eegyolk.initialization_functions import i_load_metadata
from eegyolk.initialization_functions import load_events
from eegyolk.initialization_functions import save_events
from eegyolk.initialization_functions import print_event_info
from eegyolk.initialization_functions import caller_save_events
from eegyolk.initialization_functions import generator_load_dataset

from eegyolk.data_functions import generate_frequency_distribution
from eegyolk.data_functions import create_labeled_dataset

main_path = os.path.dirname(os.getcwd())
    # repo_path = os.path.join(main_path, 'eegyolk')
repo_path = main_path
drive_path = os.path.join('tests','synthetic_data', 'parallel_testing')
    # D:\ePodium _Projectfolder
eegyolk_path = os.path.join(repo_path, 'eegyolk')
sys.path.insert(0, eegyolk_path)

dataset_path = os.path.join(drive_path, 'Dataset')
sample_eeg_bdf = os.path.join(drive_path, 'Dataset' + '/101a.bdf')
#sample_eeg_bdf = 'D:/ePodium _Projectfolder/Dataset/121a.bdf'

sample_eeg_bdf_read = mne.io.read_raw_bdf(sample_eeg_bdf, preload=True)
sample_metadata = os.path.join(drive_path, 'Metadata', 'para_cdi.txt')
event_marker_folder = os.path.join(drive_path, 'events')    
sample_eeg_list = ['101a']

class TestDisplayHelperMethods(unittest.TestCase):

    def test_make_ordinal(self):
        self.assertEqual(make_ordinal(5), '5th')


class TestFilteringMethods(unittest.TestCase):

    def test_band_pass_filter(self):
        sample_eeg_filtered = band_pass_filter(sample_eeg_bdf_read, 0, 10)
        self.assertEqual(
            (sample_eeg_filtered.info['lowpass']),
            10,
        )
    def test_filter_raw(self):
        sample_eeg_raw_filtered = filter_eeg_raw(sample_eeg_bdf_read, 0, 10,50,  ['EXG1', 'EXG2'], [])
        self.assertEqual(
            (sample_eeg_raw_filtered.info['lowpass']),
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
   
    # def test_load_metadata(self):
    #     filename = os.path.splitext(sample_metadata)[0]
    #     loaded_metadata = load_metadata(
    #         filename,
    #         sys.path[0],
    #         sys.path[0],
    #         make_excel_files=False,
    #         make_csv_files=False,
    #     )
    #     self.assertEqual(len(loaded_metadata), 14)

    def test_load_events(self):
        loaded_event_markers = load_events(event_marker_folder, sample_eeg_list)
        self.assertEqual(len(loaded_event_markers), 1)
    
    def test_call_event_markers(self):
        # temporary directory
        expected = 3
        with TemporaryDirectory() as td:
            caller_save_events(td, islice(generator_load_dataset(dataset_path), 3))
            actual = sum(1 for txt in glob.glob(os.path.join(td,'*.txt')))
        # compare number files generated,to expected which we stop at 3 with 
        self.assertEqual(expected, actual)


class TestDummyDataMethods(unittest.TestCase):

    print("-- START TEST DUMMY DATA --")

    def test_generate_frequency_distribution(self):
        max_freq = 256
        freq_sample_rate = 10
        frequency_distribution = generate_frequency_distribution(distribution = "planck",
        max_freq=max_freq,
        freq_sample_rate=freq_sample_rate,
        )
        self.assertEqual(len(frequency_distribution), max_freq * freq_sample_rate)

    def test_create_labeled_dataset(self):
        size = 5
        duration = 2
        sample_rate = 512

        labeled_dataset = create_labeled_dataset(size=size)
        self.assertEqual(labeled_dataset[0].shape,
        (size, duration * sample_rate))
        self.assertEqual(labeled_dataset[1].shape, (size, ))


if __name__ == '__main__':
    unittest.main()
