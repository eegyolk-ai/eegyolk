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


#Fourier Transformation  ------------
yf = rfft(data_fragment)
xf = rfftfreq(time_steps, 1 / sample_rate)
disp.show_plot(xf, np.abs(yf), "Fourier Transform of Data Fragment", 
               "frequency (Hz)", "Channel voltage (\u03BCV)")


































