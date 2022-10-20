from tensorflow.keras.utils import Sequence
import random

class EpodiumSequence(Sequence):
    """
    An Iterator Sequence class as input to feed the model.
    The next value is given from the __getitem__ function.
    For more information on Sequences, go to:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    self.labels contains:  ['Participant', 'Age_days_a', 'Age_days_b', 'Risk_of_dyslexia']
    """    

    def __init__(self, experiments, target_labels, epochs_directory, n_experiments_batch=8, n_trials_averaged=30, gaussian_noise=0):
        self.experiments = experiments
        self.labels = target_labels
        self.epochs_directory = epochs_directory
        
        self.n_experiments_batch = n_experiments_batch
        self.n_trials_averaged = n_trials_averaged
        self.gaussian_noise = gaussian_noise


    # The number of experiments in the entire dataset.
    def __len__(self):
        return int(np.ceil(len(self.experiments)/self.n_experiments_batch))

    def __getitem__(self, index, verbose=False):        
        x_batch = []
        y_batch = []
        
        #print(self.labels)

        for i in range(self.n_experiments_batch):

            # Set participant
            experiment_index = (index * self.n_experiments_batch + i) % len(self.experiments)
            experiment = self.experiments[experiment_index]
            participant = experiment[:3]
            participant_labels = self.labels.loc[self.labels['Participant']==float(participant)]

            if(verbose):
                print(experiment)
                
            # Load .fif file
            path_epochs = os.path.join(epochs_directory, experiment + "_epo.fif")
            epochs = mne.read_epochs(path_epochs, verbose=0)
            print(epochs)
            
            # A data instance is created for each condition
            for condition in ['GiepM', "GiepS", "GopM", "GopS"]:
                
                standard_event = condition + '_S'
                deviant_event = condition + '_D'
                standard_data = epochs[standard_event].get_data() # TODO: DDP channels/ePod
                deviant_data = epochs[deviant_event].get_data()
                                
                # Create ERP from averaging 'n_trials_averaged' trials.
                trial_indexes_S = np.random.choice(standard_data.shape[0], self.n_trials_averaged, replace=False)
                evoked_S = np.mean(standard_data[trial_indexes_S,:,:], axis=0)
                trial_indexes_D = np.random.choice(deviant_data.shape[0], self.n_trials_averaged, replace=False)
                evoked_D = np.mean(deviant_data[trial_indexes_D,:,:], axis=0)
                
                x_batch.append(evoked_S)

                ## Merge Standard and Deviant evoked along the channel dimensions.
                # evoked = np.concatenate((evoked_S, evoked_D))
                # evoked += np.random.normal(0, self.gaussian_noise, evoked.shape)
                # x_batch.append(evoked)

                # Binary labels:
                # y = np.zeros(2)
                # if participant_labels["Sex"].item() == "M" :
                #     y[0] = 1
                # if participant_labels["Group_AccToParents"].item() == "At risk":
                #     y[1] = 1
                
                # Append age to target 'y'
                if str(experiment[-1]) == "a":
                    y = int(participant_labels[f"Age_days_a"].item())
                elif str(experiment[-1]) == "b":
                    try: 
                        y = int(participant_labels[f"Age_days_b"].item())
                    except: # If age of 'b' experiment not in metadata
                        y = int(participant_labels[f"Age_days_a"].item()) + 120

                y_batch.append(y)

        # Shuffle batch
        shuffle_batch = list(zip(x_batch, y_batch))
        random.shuffle(shuffle_batch)
        x_batch, y_batch = zip(*shuffle_batch)

        return np.array(x_batch), np.array(y_batch)

    
class DDPSequence(Sequence):
    """
    An Iterator Sequence class as input to feed the model.
    The next value is given from the __getitem__ function.
    For more information on Sequences, go to:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

    self.labels contains:  ['filename', 'participant', 'age_group', 'age_days']
    """    

    def __init__(self, experiments, target_labels, epochs_directory, n_experiments_batch=8,
                 n_instances_per_experiment=4, n_trials_averaged=30, gaussian_noise=0):
        self.experiments = experiments
        self.labels = target_labels
        self.epochs_directory = epochs_directory
        
        self.n_experiments_batch = n_experiments_batch
        self.instances = n_instances_per_experiment
        self.n_trials_averaged = n_trials_averaged
        self.gaussian_noise = gaussian_noise


    # The number of experiments in the entire dataset.
    def __len__(self):
        return int(np.ceil(len(self.experiments)/self.n_experiments_batch))

    def __getitem__(self, index, verbose=False):        
        x_batch = []
        y_batch = []
        
        #print(self.labels)

        for i in range(self.n_experiments_batch):

            # Set participant
            experiment_index = (index * self.n_experiments_batch + i) % len(self.experiments)
            experiment = self.experiments[experiment_index]
                        
            participant = experiment.split("_")[0]
            age_group = experiment.split("_")[1]
                        
            experiment_labels = self.labels.loc[(self.labels['participant']==float(participant))
                                                & (self.labels['age_group']==int(age_group))]

            if(verbose):
                print(experiment)
                
            # Load .fif file
            path_epochs = os.path.join(epochs_directory, experiment + "_epo.fif")
            epochs = mne.read_epochs(path_epochs, verbose=0)
            
            # A data instance is created for each condition
            for j in range(self.instances):
                
                standard_data = epochs["standard"].get_data()
                                
                # Create standardised ERP from averaging 'n_trials_averaged' trials.
                trial_indexes_standards = np.random.choice(standard_data.shape[0], self.n_trials_averaged, replace=False)
                evoked_standard = np.mean(standard_data[trial_indexes_standards,:,:], axis=0)
                
                evoked_standard += np.random.normal(0, self.gaussian_noise, evoked_standard.shape)
                # Divide by standard deviation to make the range of signals more similar:
                evoked_standard_std = evoked_standard/evoked_standard.std()
                
                x_batch.append(evoked_standard_std)
                y_batch.append(experiment_labels["age_days"].iloc[0])


        # Shuffle batch
        shuffle_batch = list(zip(x_batch, y_batch))
        random.shuffle(shuffle_batch)
        x_batch, y_batch = zip(*shuffle_batch)

        return np.array(x_batch), np.array(y_batch)
