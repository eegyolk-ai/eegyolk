import mne
import pandas as pd 
import numpy as np 
import os           
import glob 
from IPython.display import clear_output

from eegyolk.display_helper import make_ordinal


def generator_load_dataset(folder_dataset, file_extension='.bdf', preload=True):
    """
    Documentation
    """
    pattern = os.path.join(folder_dataset, '**/*' + file_extension)
    eeg_filepaths = glob.glob(pattern, recursive=True)
    for path in eeg_filepaths:                
        bdf_file = mne.io.read_raw_bdf(path,preload=preload)
        print('file is read')
        eeg_filename = os.path.split(path)[1].replace(file_extension, '')
        yield bdf_file, eeg_filename
        # clear_output(wait=True)
    print(len(eeg_filepaths), "EEG files loaded")

def load_dataset(folder_dataset, file_extension = '.bdf', preload=True):
    '''
    This function is for datasets under 5 files. Otherwise use generator_load_dataset
    Reads and returns the bdf files that store the EEG data,
    along with a list of the filenames and paths of these bdf files. 
    Takes as input the top folder location of the dataset.
    '''
    pattern = os.path.join(folder_dataset, '**/*' + file_extension)
    eeg_filepaths = glob.glob(pattern, recursive=True)
    eeg_dataset = []
    eeg_filenames = []
    for path in eeg_filepaths:
        if(file_extension == '.bdf'):
            BdfFile = mne.io.read_raw_bdf(path,preload=preload)
            eeg_dataset.append(BdfFile)
            eeg_filenames.append(os.path.split(path)[1].replace(file_extension, ''))
        clear_output(wait=True)
    print(len(eeg_dataset), "EEG files loaded")
    return eeg_dataset, eeg_filenames


def load_metadata(folder, filenames):
    '''
    Reads and returns the four metadata text files. 
    Takes as input the folder location of the metadata files.
    '''
    metadata = []
    for filename in filenames:
        path = os.path.join(folder, filename)
        metadata.append(pd.read_table(path)) 
    return metadata


def save_event_markers(folder_event_markers, eeg_dataset, eeg_filenames):
    '''    
    Events are loaded from raw EEG files and saved in .txt file.
    Loading from .txt file is much faster than from EEG file.
    '''
    if not(os.path.exists(folder_event_markers)):
        os.mkdir(folder_event_markers)

    for i in range(len(eeg_dataset)):
        event_marker_path = os.path.join(folder_event_markers, eeg_filenames[i] + ".txt")
        np.savetxt(event_marker_path, mne.find_events(eeg_dataset[i]), fmt = '%i')
        print("\n", i+1, " out of ", len(eeg_dataset), " saved.")
        clear_output(wait=True)

def caller_save_event_markers(folder_event_markers, generator_argument):
    '''    
    Events are loaded from raw EEG files and saved in .txt file.
    Loading from .txt file is much faster than from EEG file.
    '''
    if not os.path.isdir(folder_event_markers):
        os.mkdir(folder_event_markers)

    for i, (file, filename) in enumerate(generator_argument):
        event_marker_path = os.path.join(folder_event_markers, filename + ".txt")
        np.savetxt(event_marker_path, mne.find_events(file), fmt = '%i')
        print("\n", i, " saved.")
        clear_output(wait=True)

def load_event_markers(folder_event_markers, eeg_filenames):
    '''
    Events are saved and loaded externally from .txt file, 
    since loading events from raw EEG file takes much longer. 
    NB: eeg_filenames should not include extension or root directory.
    '''
    if not(os.path.exists(folder_event_markers)):
        print("There is no folder at: ", folder_event_markers,
              "\n first save the events in this folder.")
        return None

    event_markers = []
    for filename in eeg_filenames:
        filepath = os.path.join(folder_event_markers, filename + ".txt")
        event_markers.append(np.loadtxt(filepath, dtype = int))    
    print(len(event_markers), "Event Marker files loaded")
    return event_markers


def print_event_info(event_markers, participant_index = 5, event_index = 500, sample_frequency = 2048):
    '''
    Prints information on a specified event marker.
    '''
    event_time = event_markers[participant_index][event_index][0]
    event_ID = event_markers[participant_index][event_index][2]
    print((
        "Participant {} heard event ID: {} after {:.1f} seconds "
        "as the {} event"
    ).format(
        participant_index,
        event_ID,
        event_time / sample_frequency,
        make_ordinal(event_index),
    ))
