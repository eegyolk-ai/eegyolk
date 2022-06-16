# SCRIPT TO STORE FUNCTIONALITIES FOR EASY ACCESS TO WRITE OTHER SCRIPTS

import mne              # toolbox for analyzing and visualizing EEG data
import pandas as pd     # data analysis and manipulation
import numpy as np      # numerical computing (manipulating and performing operations on arrays of data)
import os               # using operating system dependent functionality (folders)
import glob             # functions for matching and finding pathnames using patterns
import copy             # Can Copy and Deepcopy files so original file is untouched
import sys              # system-specific information and resources

main_path = os.path.dirname(os.path.dirname(os.getcwd()))
repo_path = os.path.join(main_path, 'ePodium')
data_path = os.path.join(main_path, 'researchdrive', 'ePodium (Projectfolder)')

eegyolk_path = os.path.join(repo_path, 'eegyolk')
sys.path.insert(0, eegyolk_path)
from eegyolk import initialization_functions as ifun
from eegyolk import epod_helper as epod
from eegyolk import display_helper as disp

folder_epod_dataset = os.path.join(data_path, "Dataset")
folder_epod_metadata = os.path.join(data_path, "Metadata")
folder_epod_events = os.path.join(data_path, "events")



epod_raw, epod_filenames = ifun.load_dataset(folder_epod_dataset, preload=False)


epod_metadata_filenames = ["children.txt", "cdi.txt", "parents.txt", "CODES_overview.txt"]  
epod_children, epod_cdi, epod_parents, epod_codes = ifun.load_metadata(folder_epod_metadata, epod_metadata_filenames)



fig = mne.viz.plot_events(events_12[4], event_id = epod.event_dictionary, color = disp.color_dictionary)

#### Plot of EEG signal with events

participant_index = 3
# %matplotlib qt / widget for interactive (remove 'fig = ' if no figure shows)
fig = mne.viz.plot_raw(epod_raw[participant_index], events[participant_index], n_channels=16, scalings = 50e-6  ,duration = 1, start = 500)





##### FOURIER TRANSFORMATION 
yf = rfft(data_fragment)
xf = rfftfreq(time_steps, 1 / sample_rate)
disp.show_plot(xf, np.abs(yf), "Fourier Transform of Data Fragment", "frequency (Hz)", "Channel voltage (\u03BCV)")


































