import mne              # toolbox for analyzing and visualizing EEG data
import pandas as pd     # data analysis and manipulation
import numpy as np      # numerical computing (manipulating and performing operations on arrays of data)
import os               # using operating system dependent functionality (folders)
import glob             # functions for matching and finding pathnames using patterns
import copy             # Can Copy and Deepcopy files so original file is untouched
from IPython.display import clear_output


def load_dataset(folder_dataset):
    '''
    Reads and returns the bdf files that store the EEG data,
    along with a list of the filenames and paths of these bdf files. 
    Takes as input the folder location of the dataset.
    '''
    pattern = os.path.join(folder_dataset, '**/*.bdf')
    eeg_filepaths = glob.glob(pattern, recursive=True)
    eeg_dataset = []
    eeg_filenames = []
    for path in eeg_filepaths:
        BdfFile = mne.io.read_raw_bdf(path)
        eeg_dataset.append(BdfFile)
        eeg_filenames.append(os.path.split(path)[1].replace(".bdf", ""))
        clear_output(wait=True)
    print(len(eeg_dataset), "EEG files loaded")
    return eeg_dataset, eeg_filenames, eeg_filepaths


def load_metadata(folder_metadata):
    '''
    Reads and returns the four metadata text files. 
    Takes as input the folder location of the metadata files.
    '''
    metadata_filenames = ["children.txt", "cdi.txt", "parents.txt", "CODES_overview.txt"]      
    metadata = []
    for filename in metadata_filenames:
        path = os.path.join(folder_metadata, filename)
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
        print("\n", i+1, " out of ", len(eeg_dataset), " loaded.")
        clear_output(wait=True)


def load_event_markers(folder_event_markers, eeg_filenames):
    '''
    Events are saved and loaded externally from .txt file, 
    since loading events from raw EEG file takes much longer. 
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
    return event_markers;


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


def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def group_event_markers(event_markers):
    '''
    Reduces the number of distinctive events from 78 to 12 events.
    This is done by combining different pronounciations into the same event.
    '''    
    event_markers_12 = copy.deepcopy(event_markers)
    for i in range(len(event_markers)):
        for newValue, minOld, maxOld in event_conversion_12:
            condition = np.logical_and(minOld <= event_markers_12[i], event_markers_12[i] <= maxOld)
            event_markers_12[i] = np.where(condition, newValue, event_markers_12[i])
    return event_markers_12

event_dictionary = {
    'GiepMT_FS': 1,
    'GiepMT_S': 2,
    'GiepMT_D': 3,
    'GiepST_FS': 4,
    'GiepST_S': 5,
    'GiepST_D': 6,
    'GopMT_FS': 7,
    'GopMT_S': 8,
    'GopMT_D': 9,
    'GopST_FS': 10,
    'GopST_S': 11,
    'GopST_D': 12,
}

event_conversion_12 = [
    [1, 1, 12], 
    [2, 13, 24], 
    [3, 25, 36], 
    [4, 101, 101], 
    [5, 102, 102], 
    [6, 103, 103],
    [7, 37, 48],
    [8, 49, 60],
    [9, 61, 72],
    [10, 104, 104],
    [11, 105, 105], 
    [12, 106, 106]
]

color_dictionary = {
    1: "#8b0000",
    2: "#008000",
    3: "#000080",
    4: "#ff0000",
    5: "#ff1493",
    6: "#911eb4",
    7: "#87cefa",
    8: "#ffd700",
    9: "#696969",
    10: "#000000", 
    11: "#1e90ff",
    12: "#7fff00",
}

def tester():
    print("test complete")







