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
