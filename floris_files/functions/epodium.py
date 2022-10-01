"""Tools for working with the ePodium dataset."""

import mne
import numpy as np
import pandas as pd
import copy
import os
import glob
from IPython.display import clear_output

# This class is useful for passing the dataset object as input to a function.
class Epodium:
    
    ############ --- GENERAL DATASET TOOLS --- #################
    
    file_extension = ".bdf"
    frequency = 2048 # Hz

    metadata_filenames = ["children.txt", "cdi.txt", "parents.txt", "CODES_overview.txt"]

    channel_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5',
                   'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
                   'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
                   'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
                   'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
                   'Fz', 'Cz']

    channels_mastoid = ['EXG1', 'EXG2']
    channels_drop = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
    
    montage = 'standard_1020'
    mne_montage = mne.channels.make_standard_montage(montage) 
    mne_info = mne.create_info(channel_names, frequency, ch_types='eeg')

    conditions = ['GiepM', "GiepS", "GopM", "GopS"]
    conditions_events = ['GiepM_S', 'GiepM_D', 'GiepS_S', 'GiepS_D', 'GopM_S', 'GopM_D', 'GopS_S', 'GopS_D']

    event_dictionary = {'GiepM_FS': 1, 'GiepM_S': 2, 'GiepM_D': 3, 
                        'GiepS_FS': 4, 'GiepS_S': 5, 'GiepS_D': 6,
                        'GopM_FS': 7,  'GopM_S': 8,   'GopM_D': 9,
                        'GopS_FS': 10, 'GopS_S': 11,  'GopS_D': 12, }   
    
    # To be ignored during processing: 
    incomplete_experiments = ["113a", "107b (deel 1+2)", "132a", "121b(2)", "113b", "107b (deel 3+4)", "147a",
                              "121a", "134a", "143b", "121b(1)", "145b", "152a", "184a", "165a", "151a", "163a",
                              "207a", "215b"]    

    
    def get_events_from_raw(self, raw):
        events = mne.find_events(raw, verbose=False, min_duration=2/self.frequency)
        events_12 = self.group_events_12(events)
        return events_12
    
    
    ############ --- TOOLS SPECIFICCALLY FOR THE EPODIUM DATASET --- #################

    ############ --- LOADING --- #################
    @staticmethod
    def load_events(folder_events, eeg_filenames):
        if not(os.path.exists(folder_events)):
            print("There is no folder at: ", folder_events,
                  "\n first save the events in this folder.")
            return None

        events = []
        for filename in eeg_filenames:
            filepath = os.path.join(folder_events, filename + ".txt")
            events.append(np.loadtxt(filepath, dtype=int))
        print(len(events), "Event Marker files loaded")
        return events

    @staticmethod
    def load_dataset(folder_dataset, file_extension='.bdf', preload=False):
        pattern = os.path.join(folder_dataset, '**/*' + file_extension)
        eeg_filepaths = sorted(glob.glob(pattern, recursive=True))
        eeg_dataset = []
        eeg_filenames = []

        for i, path in enumerate(eeg_filepaths):
            raw = mne.io.read_raw_bdf(path, preload=preload)
            eeg_dataset.append(raw)
            filename = os.path.split(path)[1].replace(file_extension, '')
            eeg_filenames.append(filename)
            print(i+1, "EEG files loaded")

            clear_output(wait=True)
        print(i+1, "EEG files loaded")

        return eeg_dataset, eeg_filenames

    def load_metadata(self, folder):
        metadata = []
        for filename in self.metadata_filenames:
            path = os.path.join(folder, filename)
            metadata.append(pd.read_table(path))
        return metadata  

    ############ --- EVENTS --- #################
    
    @staticmethod
    def group_events_12(events):
        """
        Puts similar event conditions with different pronounciations together
        Reduces the number of distinctive events from 78 to 12 events.
        This is done by combining different pronounciations into the same event.
        """
        event_conversion_12 = [
            [1, 1, 12], [2, 13, 24], [3, 25, 36],
            [4, 101, 101], [5, 102, 102], [6, 103, 103],
            [7, 37, 48], [8, 49, 60], [9, 61, 72],
            [10, 104, 104], [11, 105, 105], [12, 106, 106]]

        events_12 = copy.deepcopy(events)
        for i in range(len(events)):
            for newValue, minOld, maxOld in event_conversion_12:
                condition = np.logical_and(
                    minOld <= events_12[i], events_12[i] <= maxOld)
                events_12[i] = np.where(condition, newValue, events_12[i])
        return events_12

    @staticmethod
    def save_events(folder_events, eeg_dataset, eeg_filenames):
        """
        This function loads the events from the raw file and saves them in an external folder.
        Loading from a .txt file is many times faster than loading from raw.
        """
        if not os.path.exists(folder_events):
            os.mkdir(folder_events)

        for i in range(len(eeg_dataset)):
            path_events = os.path.join(folder_events, eeg_filenames[i] + ".txt")
            if(os.path.exists(path_events)):
                print(f"Event .txt file for {eeg_filenames[i]} already exists")
            else:
                np.savetxt(path_events, mne.find_events(eeg_dataset[i], min_duration = 2/frequency), fmt='%i')
                print("\n", i + 1, " out of ", len(eeg_dataset), " saved.")                
            clear_output(wait=True)



    ############ --- PLOTTING --- #################

    @staticmethod
    def plot_ERP(epochs, condition, event_type, save_path = ""):
        standard = condition + "_S"
        deviant = condition + "_D"

        if(event_type == "standard"):
            evoked = epochs[standard].average()    
        elif(event_type == "deviant"):
            evoked = epochs[deviant].average()    
        elif(event_type == "MMN"):
            evoked = mne.combine_evoked([epochs[deviant].average(), epochs[standard].average()], weights = [1, -1])

        fig = evoked.plot(spatial_colors = True)

        if save_path:
            fig.savefig(save_path)





############ --- DEPRECATED --- #################

# Now using the mne .fif file instead of numpy's .npy (more user-friendly and data efficient)
def load_cleaned_npy_file(path_npy, path_events):
    npy = np.load(path_npy)
    events_12 = np.loadtxt(path_events, dtype=int)

    epochs = mne.EpochsArray(npy, mne_info, events=events_12, tmin=-0.2,
                             event_id=event_dictionary, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.info.set_montage(montage, on_missing = 'ignore')

    print(f"{npy.shape[0]} different trials loaded \nEach trial contains {npy.shape[2]} timesteps in {npy.shape[1]} channels.")    
    return epochs


channel_dict = {'Fp1':'eeg', 'AF3':'eeg', 'F7':'eeg', 'F3':'eeg', 'FC1':'eeg',
                 'FC5':'eeg', 'T7':'eeg', 'C3':'eeg', 'CP1':'eeg', 'CP5':'eeg',
                 'P7':'eeg', 'P3':'eeg', 'Pz':'eeg', 'PO3':'eeg', 'O1':'eeg', 
                 'Oz':'eeg', 'O2':'eeg', 'PO4':'eeg', 'P4':'eeg', 'P8':'eeg',
                 'CP6':'eeg', 'CP2':'eeg', 'C4':'eeg', 'T8':'eeg', 'FC6':'eeg',
                 'FC2':'eeg', 'F4':'eeg', 'F8':'eeg', 'AF4':'eeg', 'Fp2':'eeg',
                 'Fz':'eeg', 'Cz':'eeg'}


def split_downsample_clean_epochs(cleaning_method, sample_rate=128):
    """
        This function splits the cleaned epochs up into into a seperate file for each event.
        The sampling rate is also reduced to decrease the data size.

        From path_clean, the function uses the epochs from the 'epochs' folder and saves them in 'epochs_split'.
    """

    path_cleaned = os.path.join(local_paths.epod_clean, "ePod_" + cleaning_method)
    path_split = os.path.join(local_paths.epod_split, cleaning_method + "_" + str(sample_rate) + "hz")
    if not os.path.exists(path_split): os.mkdir(path_split)

    montage = mne.channels.make_standard_montage('standard_1020') 
    info = mne.create_info(epodium.channel_names, 2048, ch_types='eeg')

    npy_filepaths = glob.glob(os.path.join(path_cleaned, 'epochs', '*.npy'))
    for npy_filepath in npy_filepaths:
        npy_filename = os.path.basename(npy_filepath)
        filename = os.path.splitext(npy_filename)[0]

        # Find missing files
        missing_split_paths = []
        for event in epodium.event_dictionary:
            split_filename = filename + "_" + event + ".npy"
            path_split_file = os.path.join(path_split, split_filename)

            if not os.path.exists(path_split_file):
                missing_split_paths.append(path_split_file)

        # Split and save missing files
        if missing_split_paths != []:
            npy = np.load(os.path.join(path_cleaned, 'epochs', npy_filepath))
            events_12 = np.loadtxt(os.path.join(path_cleaned, 'events', filename + ".txt"), dtype=int)
            epochs = mne.EpochsArray(npy, info, events=events_12, tmin=-0.2, 
                                     event_id=epodium.event_dictionary, verbose=False)
            epochs.info.set_montage(montage, on_missing='ignore')

            for missing_split_file in missing_split_paths: 
                np.save(missing_split_file, epochs[event].resample(sample_rate).get_data())
                print(f"{os.path.basename(missing_split_file)} saved")

                
def load_epochs_from_npy_and_events(path_npy, path_events):
    npy = np.load(path_npy)
    events_12 = np.loadtxt(path_events, dtype=int)

    epochs = mne.EpochsArray(npy, epodium.mne_info, events=events_12, tmin=-0.2,
                             event_id=epodium.event_dictionary, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.info.set_montage(montage, on_missing = 'ignore')

    print(f"{npy.shape[0]} different trials loaded \nEach trial contains {npy.shape[2]} timesteps in {npy.shape[1]} channels.")    
    return epochs

def load_participant_data(experiment):
    path_cleaned = os.path.join(local_paths.epod_clean, "ePod_" + processing)
    path_npy = os.path.join(path_cleaned, "epochs",  experiment + ".npy")
    path_events = os.path.join(path_cleaned, "events", experiment + ".txt")
    epochs = load_epochs_from_npy_and_events(path_npy, path_events)
    return epochs


# SAVE ERPs FROM CLEAN EPOCHS
def save_erp_figures_from_epochs():
    processing = "ransac"
    save_path = "C:\Floris\Python Folder\Thesis Code"
    path_cleaned_epochs = [x for x in os.path.join(local_paths.epod_clean, "ePod_" + processing)]

    experiments = [f[0:4] for f in os.listdir(os.path.join(local_paths.epod_clean, "ePod_" + processing, "epochs")) if f.endswith(".npy")]
    for experiment in experiments:
        epochs = load_participant_data(experiment)

        for condition in epodium.conditions:
            for s_d in ["_S", "_D"]:
                condition_type = condition + s_d
                path = os.path.join(local_paths.epod_personal, "results", experiment+ "_" + condition_type+"_"+processing+".png")
                if not os.path.exists(path):
                    try:
                        evoked = epochs[condition_type].average()
                        fig = evoked.plot(spatial_colors = True)
                        fig.savefig(path)
                    except:
                        print(experiment + condition_type)
