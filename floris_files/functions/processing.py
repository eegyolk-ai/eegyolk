import mne
import numpy as np
import autoreject
import os
import glob

## Multiprocessing
import multiprocessing
import functools


def process_raw_multiprocess(raw_paths, dataset, processed_directory, num_processes = 8, verbose=False):
    with multiprocessing.Pool(num_processes) as pool:
        # Set fixed input variables
        partial = functools.partial(process_raw, dataset=dataset, processed_directory=processed_directory, verbose=verbose)
        # Multiprocess with respect to raw paths
        pool.map(partial, raw_paths)
    
    print("All files processed")

    
def process_raw(raw_path, dataset, processed_directory, verbose=False):
    """
        This function processes a raw EEG file from the MNE class.
        Processing steps: 1. high-pass filter, 2. create epochs, 3. low-pass filter, 4. AutoReject.
        The processed .fif file is stored in processed_directory.
        This directory should contain an 'events' folder to store the events .txt file.
        
        Args:
        raw_path: Path to the raw EEG-file
        dataset: Class containing information on the dataset, e.g. 
        processed_directory: Directory for storing the files.
    """    
    
    # Raw file-names
    file = os.path.basename(raw_path)
    filename, extension = os.path.splitext(file)
    
    # Paths for cleaned data    
    path_epoch = os.path.join(processed_directory, filename+"_epo.fif")
    path_events = os.path.join(processed_directory, "events", filename+"_events.txt")  

    # Check if file needs to be processed:
    if os.path.exists(path_epoch) or os.path.exists(path_events):
        if verbose:
            print(f"File {file} already cleaned \n", end='')
        # If the event .txt file is missing:
        if not os.path.exists(path_events):
            print(f"Creating the event file {filename}.txt \n", end='')
            epochs_clean = mne.read_epochs(path_epoch, verbose=0)
            np.savetxt(path_events, epochs_clean.events, fmt='%i')
        return    
    if filename in dataset.incomplete_experiments:
        if verbose:
            print(f"File {file} ignored \n", end='')
        return
    if verbose:
        print(f"Cleaning file: {file}  \n" , end='')
    
    
    # Read-in raw file
    raw = dataset.read_raw(raw_path)    
    events, event_dict = dataset.get_events_from_raw(raw)

    # Set electrodes
    raw.pick_channels(dataset.channel_names)
    raw.info.set_montage(dataset.mne_montage, on_missing='ignore')

    # High-pass filter for detrending
    raw.filter(0.1, None, verbose=False)
    
    # Create epochs from raw. Epoch creation sometimes returns an error.
    try:
        epochs = mne.Epochs(raw, events, event_dict, -0.2, 0.8, preload=True, verbose=False)
    except:
        print(f"Not all events of the event_dictionary in file {file} \n", end='')
        return
    
    # Low pass filter for high-frequency artifacts
    epochs.filter(None, 40, verbose=False)

    # Reject bad trials and repair bad sensors in EEG
    # autoreject.Ransac() is a quicker but less accurate method than AutoReject.
    ar = autoreject.AutoReject() 
    epochs_clean = ar.fit_transform(epochs)  

    # # Save data and events    
    epochs_clean.save(path_epoch)
    # np.save(path_cleaned_file, epochs_clean.get_data())   
    np.savetxt(path_events, epochs_clean.events, fmt='%i')
    
    
def valid_experiments(event_directory, min_standards = 180, min_deviants = 80, min_firststandards = 0):
    """
    This function checks the number of remaining epochs in the event files after cleaning.
    In an ideal epodium experiment, there are 360 standards, 120 deviants and 130 first standards in each of the 4 conditions.    
    """
    
    # ePodium setup of the 12 events in 4 conditions.
    firststandard_index = [1, 4, 7, 10]
    standard_index = [2, 5, 8, 11]
    deviant_index = [3, 6, 9, 12]

    # Experiments with enough epochs are added to valid_experiments
    valid_experiments = []
    
    paths_events = glob.glob(os.path.join(event_directory, '*.txt'))
    for path_events in paths_events:
        event = np.loadtxt(path_events, dtype=int)
        
        # Counts how many events are left in standard, deviant, and FS in the 4 conditions.
        for i in range(4):
            if np.count_nonzero(event[:, 2] == standard_index[i]) < min_standards\
            or np.count_nonzero(event[:, 2] == deviant_index[i]) < min_deviants\
            or np.count_nonzero(event[:, 2] == firststandard_index[i]) < min_firststandards:
                break
        else: # No bads found at end of for loop
            filename_event = os.path.basename(path_events).split(("_"))[0]
            valid_experiments.append(filename_event)

    valid_experiments = sorted(valid_experiments)
    print(f"Analyzed: {len(paths_events)}, bad: {len(paths_events) - len(valid_experiments)}")
    print(f"{len(valid_experiments)} experiments have enough epochs for analysis.")
    
    return valid_experiments