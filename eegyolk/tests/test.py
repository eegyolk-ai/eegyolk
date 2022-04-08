# testing eegyolk

import unittest
import os
import glob
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0, os.getcwd())

from eegyolk.display_helper import make_ordinal
from eegyolk.helper_functions import band_pass_filter
from eegyolk.helper_functions import hash_it_up_right_all

sample_eeg = 'sample/sample_124a_cut.bdf'
# are you sure


            
        

class TestDisplayHelperMethods(unittest.TestCase):

    def test_make_ordinal(self):
        self.assertEqual(make_ordinal(5), '5th')

# need to test filtering methods- needs thought.
# class TestFilteringMethods(unittest.TestCase):

    def test_band_pass_filter(self):
        # need to hash for contents then assert equal
        sample_eeg_filtered = band_pass_filter(sample_eeg, 20, 21)
        self.assertFalse(
            band_pass_filter(sample_eeg).equals(sample_eeg_filtered)
        )


# # need a test for helper_functions.hash_it_up_right_all(), test currently failing
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

if __name__ == '__main__':
    unittest.main()

