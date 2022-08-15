"""
Copyright 2022 Netherlands eScience Center and Utrecht University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to deal with inconsistencies
that may appear in certain formats of EEG files.
"""

from datetime import datetime
from math import modf

import mne.io.cnt._utils as mne_cnt_utils
import mne.io.cnt.cnt as mne_cnt_cnt

from mne.utils import warn


def _session_date_2_meas_date(session_date, date_format):
    try:
        frac_part, int_part = modf(
            datetime
            .strptime(session_date, date_format)
            .timestamp()
        )
    except ValueError:
        try:
            date, time = session_date.split()
            d, m, y = date.split('/')
            if len(y) == 3:
                y = y[1:]
            else:
                raise ValueError('Didn\'t guess the date format')
            # Interestingly, whenever "FAST" date format is used, the
            # month and day are swapped...
            session_date = f'{m}/{d}/{y} {time}'
            frac_part, int_part = modf(
                datetime
                .strptime(session_date, date_format)
                .timestamp()
            )
        except ValueError:
            warn('Could not parse meas date from the header. Setting to None.')
            return None
    else:
        return int_part, frac_part


def patch():
    mne_cnt_utils._session_date_2_meas_date = _session_date_2_meas_date
    mne_cnt_cnt._session_date_2_meas_date = _session_date_2_meas_date
