class EpodiumSequence(Sequence):
    """
        An Iterator Sequence class as input to feed the model.
        The next value is given from the __getitem__ function.
        For more information on Sequences, go to:
        https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

        self.experiments
        self.metadata contains:
    """    

    def __init__(self, experiments, epochs_directory, metadata_df, target_label, n_experiments_batch = 8, n_trials_averaged = 30, gaussian_noise = 0):
        self.experiments = experiments
        self.n_experiments_batch = n_experiments_batch
        self.n_trials_averaged = n_trials_averaged
        self.gaussian_noise = gaussian_noise

        # metadata_path = os.path.join(local_paths.ePod_metadata, "children.txt")
        self.metadata = pd.read_table(metadata_path)

        self.clean_path = clean_path

    # The number of experiments in the entire dataset.
    def __len__(self):
        return int(np.ceil(len(self.experiments)/self.n_experiments_batch))

    def __getitem__(self, index, verbose = False):        
        x_batch = []
        y_batch = []

        for i in range(self.n_experiments_batch):

            # Set participant
            participant_index = (index * self.n_experiments_batch + i) % len(self.experiments)
            participant = self.experiments[participant_index]
            participant_id = participant[:3]
            participant_metadata = self.metadata.loc[self.metadata['ParticipantID'] == float(participant_id)]

            if(verbose):
                print(participant)

            for condition in epodium.conditions:

                # Get Standard and Deviant file
                # npy_name_S = f"{self.experiments[participant_index]}_{condition}_S.npy"
                # npy_name_D = f"{self.experiments[participant_index]}_{condition}_D.npy"
                # npy_path_S = os.path.join(self.split_path, npy_name_S)
                # npy_path_D = os.path.join(self.split_path, npy_name_D)
                # npy_S = np.load(npy_path_S)
                # npy_D = np.load(npy_path_D)

                # Create ERP from averaging 'n_trials_averaged' trials.
                trial_indexes_S = np.random.choice(npy_S.shape[0], self.n_trials_averaged, replace=False)
                evoked_S = np.mean(npy_S[trial_indexes_S,:,:], axis=0)
                trial_indexes_D = np.random.choice(npy_D.shape[0], self.n_trials_averaged, replace=False)
                evoked_D = np.mean(npy_D[trial_indexes_D,:,:], axis=0)

                # Merge Standard and Deviant evoked along the channel dimensions.
                evoked = np.concatenate((evoked_S, evoked_D))
                evoked += np.random.normal(0, self.gaussian_noise, evoked.shape)
                x_batch.append(evoked)

                # Binary labels:
                # y = np.zeros(2)
                # if participant_metadata["Sex"].item() == "M" :
                #     y[0] = 1
                # if participant_metadata["Group_AccToParents"].item() == "At risk":
                #     y[1] = 1

                if str(participant[-1]) == "a":
                    y = normalize_age(int(participant_metadata[f"Age_days_a"].item()))
                if str(participant[-1]) == "b":
                    try: y = normalize_age(int(participant_metadata[f"Age_days_b"].item())) # Not all ages in metadata
                    except:  y = normalize_age(int(participant_metadata[f"Age_days_a"].item()) + 120)

                y_batch.append(y)

        # Shuffle batch
        shuffle_batch = list(zip(x_batch, y_batch))
        random.shuffle(shuffle_batch)
        x_batch, y_batch = zip(*shuffle_batch)

        return np.array(x_batch), np.array(y_batch)
    