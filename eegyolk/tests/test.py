# testing eegyolk

import unittest
import os
import glob
import sys
sys.path.insert(0, os.getcwd())

from eegyolk.display_helper import make_ordinal
from eegyolk.helper_functions import band_pass_filter
from eegyolk.helper_functions import hash_it_up_right_all

sample_eeg = 'sample/sample_124a.bdf'
# are you sure

class TestDisplayHelperMethods(unittest.TestCase):

    def test_make_ordinal(self):
        self.assertEqual(make_ordinal(5), '5th')

# need to test filtering methods- needs thought.
# class TestFilteringMethods(unittest.TestCase):

    # def test_band_pass_filter(self):
    #     # need to hash for contents then assert equal
    #     self.assertEqual(
    #         band_pass_filter(sample_EEG),
    #         sample_EEG_filtered,
    #     )


# # need a test for helper_functions.hash_it_up_right_all(), test currently failing
# class TestHashMethods(unittest.TestCase):

#     def test_hash_it_up_right_all(self):
#         self.assertEqual(
#             hash_it_up_right_all('sample', '.bdf'), 
#             hash_it_up_right_all('sample', '.bdf'),
#         )


if __name__ == '__main__':
    unittest.main()

