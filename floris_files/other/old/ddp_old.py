def load_dataset(folder_dataset, file_extension='.cnt', preload=False, max_files=9999, data_format='int16'):
    pattern = os.path.join(folder_dataset, '**/*' + file_extension)
    eeg_filepaths = glob.glob(pattern, recursive=True)
    eeg_dataset = []
    eeg_filenames = []
    eeg_filenames_failed_to_load = []

    files_loaded = 0
    files_failed_to_load = 0
    for path in eeg_filepaths:
        filename = os.path.split(path)[1].replace(file_extension, '')

        if file_extension == '.cnt':  # .cnt files do not always load.
            try:
                raw = mne.io.read_raw_cnt(path, preload=preload, data_format=data_format)
            except Exception:
                eeg_filenames_failed_to_load.append(filename)
                files_failed_to_load += 1
                print(f"File {filename} could not be loaded.")
                continue

        eeg_dataset.append(raw)
        eeg_filenames.append(filename)
        files_loaded += 1
        print(files_loaded, "EEG files loaded")
        clear_output(wait=True)

        if files_loaded >= max_files: 
            break

    print(len(eeg_dataset), "EEG files loaded")
    if files_failed_to_load > 0:
        print(files_failed_to_load, "EEG files failed to load")

    return eeg_dataset, eeg_filenames


def create_labels(PATH, filename):

    # 5 months is excluded from analysis, since the data is too messy (see Bjorn's thesis)
    age_groups = [11, 17, 23, 29, 35, 41, 47]

    df_list = []
    # Store cnt file info
    for age_group in age_groups:
        folder = os.path.join(PATH, str(age_group) + "mnd mmn")
        code_list = []
        path_list = []     
        file_list = []

        for file in sorted(os.listdir(folder)):
            if file.endswith(".cnt"):  
                code_list.append(int(file[0:3])) # First 3 numbers of file is participant code
                path_list.append(os.path.join(folder, file))
                file_list.append(file)

        df = pd.DataFrame({"code": code_list, "path": path_list, "file": file_list})
        df['age_group'] = age_group
        df_list.append(df)

    cnt_df = pd.concat(df_list)  

    # Set correct age labels for each file
    PATH_metadata = os.path.join(PATH , 'metadata')
    df_list = []
    for age_group in age_groups:
        age_file = "ages_" + str(age_group) + "mnths.txt"
        df = pd.read_csv(os.path.join(PATH_metadata , 'ages', age_file), sep = "\t")
        df['age_group'] = age_group
        df_list.append(df)

    age_df = pd.concat(df_list)
    age_df = age_df.drop(columns=['age_months', 'age_years']) # age_days is sufficient
    merged_df = pd.merge(cnt_df, age_df, how = 'left', on = ['age_group', 'code'])
    merged_df['age_days'].fillna(merged_df['age_group'] * 30, inplace = True)
    merged_df.to_excel(os.path.join(PATH, filename), index = True)

    
def create_labels_processed(PATH_data, PATH_labels, labels):
    # Storing each column seperately, before concatinating as DataFrame
    code_list = []
    path_list = []
    file_list = []
    age_group_list = []
    age_days_list = []

    files_path = glob.glob(PATH_data + '/*.npy')
    for file_path in files_path:
        filename = os.path.basename(os.path.splitext(file_path)[0])  
        data = labels.loc[labels['file'] == filename]
        code_list.append(data['code'].values[0] )    
        path_list.append(file_path)
        file_list.append(filename)
        age_group_list.append(data['age_group'].values[0])
        age_days_list.append(data['age_days'].values[0])

    labels_processed = pd.DataFrame({"code": code_list, 'path': path_list, "file": file_list, 
                                        'age_group': age_group_list, 'age_days': age_days_list})
    labels_processed.to_excel(PATH_labels, index = True)
    return labels_processed
