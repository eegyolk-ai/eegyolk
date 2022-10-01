import mne
import numpy as np
import autoreject
import os

def process_raw(raw_path, dataset_info, processed_directory, verbose=False):
    """
        This function processes a raw EEG file from the MNE class.
        Processing steps: 1. high-pass filter, 2. create epochs, 3. low-pass filter, 4. AutoReject.
        The processed .fif file is stored in processed_directory.
        This directory should contain an 'events' folder to store the events .txt file.
        
        Args:
        raw_path: Path to the raw EEG-file
        dataset_info: Class containing information on the dataset, e.g. 
        processed_directory: Directory for storing the files.
    """    
    
    # Raw file-names
    file = os.path.basename(raw_path)
    filename, extension = os.path.splitext(file)
    
    # Paths for cleaned data    
    path_epoch = os.path.join(processed_directory, filename+"_epo.fif")
    path_events = os.path.join(processed_directory, "events", filename+"_events.txt")  

    # If file already processed:
    if os.path.exists(path_epoch) or os.path.exists(path_events):
        if verbose:
            print(f"File {file} already cleaned \n", end='')
        # If the event .txt file is missing:
        if not os.path.exists(path_events):
            print(f"Creating the event file {filename}.txt \n", end='')
            epochs_clean = mne.read_epochs(path_epoch, verbose=0)
            np.savetxt(path_events, epochs_clean.events, fmt='%i')
        return
    
    if filename in dataset_info.incomplete_experiments:
        if verbose:
            print(f"File {file} ignored \n", end='')
        return

    if verbose:
        print(f"Cleaning file: {file}  \n" , end='')
    
    # Read-in raw file
    if extension == ".bdf":
        raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose=False)
    elif extension == ".cnt":
        raw = mne.io.read_raw_cnt(raw_path, preload=True, verbose=False)
    else:
        print(f"The file {raw_path} has doesn't exist or has an incompatible extension.")
    
    events = dataset_info.get_events_from_raw(raw)

    # Set electrodes
    raw.pick_channels(dataset_info.channel_names)
    raw.info.set_montage(dataset_info.mne_montage, on_missing='ignore')

    # High-pass filter for detrending
    raw.filter(0.1, None, verbose=False)
    
    # Create epochs from raw. Epoch creation sometimes returns an error.
    try:
        epochs = mne.Epochs(raw, events, dataset_info.event_dictionary, -0.2, 0.8, preload=True, verbose=False)
    except:
        print(f"Not all events in file {file} \n", end='')
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

    
