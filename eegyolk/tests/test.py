# testing eegyolk

import unittest
import os
import sys
sys.path.insert(0, os.getcwd())

from eegyolk.display_helper import make_ordinal
from eegyolk.helper_functions import band_pass_filter


class TestDisplayHelperMethods(unittest.TestCase):

    def test_make_ordinal(self):
        self.assertEqual(make_ordinal(5), '5th')


# class TestFiltering(unittest.TestCase):

#     def test_band_pass_filter(self):
#         # need to hash for contents then assert equal
#         self.assertEqual(band_pass_filter(sample_EEG),sample_EEG_filtered)



# need a test for helper_functions.hash_it_up_right_all()


if __name__ == '__main__':
    unittest.main()

