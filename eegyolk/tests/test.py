# testing eegyolk

import unittest
import os
import sys
sys.path.insert(0, os.getcwd())

from eegyolk.display_helper import make_ordinal


class TestDisplayHelperMethods(unittest.TestCase):

    def test_make_ordinal(self):
        self.assertEqual(make_ordinal(5), '5th')


if __name__ == '__main__':
    unittest.main()