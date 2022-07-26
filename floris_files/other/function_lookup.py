""" SCRIPT TO STORE FUNCTIONALITIES FOR EASY ACCESS TO WRITE OTHER SCRIPTS """

# Initialization -----------------

# toolbox for analyzing and visualizing EEG data
import mne
# data analysis and manipulation
import pandas as pd
# numerical computing (manipulating and performing operations on arrays of data)
import numpy as np    
# using operating system dependent functionality (folders)  
import os           
# functions for matching and finding pathnames using patterns    
import glob   
# Can Copy and Deepcopy files so original file is untouched          
import copy     
# system-specific information and resources
import sys   

# A file containing all necessary paths
import PATH

# Import in-house scripts
from functions import display_helper
from functions import dataset_loading
from functions import epodium



%matplotlib qt
mne.viz.plot_raw(raw, events)




#Fourier Transformation  ------------
yf = rfft(data_fragment)
xf = rfftfreq(time_steps, 1 / sample_rate)
disp.show_plot(xf, np.abs(yf), "Fourier Transform of Data Fragment", 
               "frequency (Hz)", "Channel voltage (\u03BCV)")


































