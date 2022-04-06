### RUN EEG data tests

#%%
# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import helper_functions 

ROOT = "C:\\OneDrive - Netherlands eScience Center\\Project_ePodium\\"
PATH_CODE = ROOT + "EEG_explorer\\"
PATH_DATA = ROOT + "Data\\"
PATH_OUTPUT = ROOT + "Data\\"


#%%
import fnmatch
dirs = os.listdir(PATH_DATA)
cnt_files = fnmatch.filter(dirs, "*.cnt")

import warnings
warnings.filterwarnings('ignore')

for filename in cnt_files[4:5]:
    # Import data and events
    file = PATH_DATA + filename
    data_raw = mne.io.read_raw_cnt(file, montage=None, eog='auto', preload=True)
    
    # Band-pass filter (between 1 and 40 Hz. Was 0.5 to 30Hz at Stober 2016)
    data_raw.filter(1, 40, fir_design='firwin')
    
    events = mne.find_events(data_raw, shortest_event=0, stim_channel='STI 014', verbose=False)
    
    event_id = [3, 13, 66] # select events for the given event IDs
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.8  # end of each epoch (500ms after the trigger)

    baseline = (None, 0)  # means from the first instant to t = 0
    picks = mne.pick_types(data_raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=True, verbose=False)

    evoked = epochs['66'].average()
    evoked.plot();
    

#%%
suspects, suspects_names = helper_functions.select_bad_channels(data_raw, 60, threshold=5)

#%% FILTERING
# 1) remove bad channels 

data_raw.info['bads'] = suspects_names

picks = mne.pick_types(data_raw.info, meg=False, eeg=True, stim=False, eog=True,
                       exclude='bads')
epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=baseline, preload=True, verbose=False)
evoked = epochs['66'].average()
evoked.plot();

#%%
# 2) detect eog based artefacts...
epochs.drop_bad(reject=dict(eog=200e-6), flat=None)
evoked = epochs['66'].average()
evoked.plot();

#%%
sfreq = data_raw.info['sfreq']
data, times = data_raw[:, int(sfreq * 100):int(sfreq * (200))]

fig = plt.subplots(figsize=(10,8))
plt.plot(times, data[30:32,:].T)
plt.xlabel('Seconds')
plt.ylabel('$\mu V$')
plt.title('Channels: 1-5')
plt.legend(data_raw.ch_names[30:32])


#%% Display epochs (averaged) for a number of patients  
for filename in cnt_files[0:6]:
    # Import data and events
    file = PATH_DATA + filename
    data_raw = mne.io.read_raw_cnt(file, montage=None, eog='auto', preload=True)
    
    # Band-pass filter (between 1 and 40 Hz. Was 0.5 to 30Hz at Stober 2016)
    #data_raw.filter(0.5, None, fir_design='firwin') #high-pass filter
    #data_raw.filter(0.5, 30, fir_design='firwin')  # Band-pass filter
    data_raw.filter(0.1, 30, fir_design='firwin')  # Band-pass filter
    
    events = mne.find_events(data_raw, shortest_event=0, stim_channel='STI 014', verbose=False)
    
    event_id = [3, 13, 66] # select events for the given event IDs
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.8  # end of each epoch (500ms after the trigger)

    baseline = (None, 0)  # means from the first instant to t = 0
    # 1) remove bad channels
    suspects, suspects_names = helper_functions.select_bad_channels(data_raw, 60, threshold=5)
    data_raw.info['bads'] = suspects_names
    
    picks = mne.pick_types(data_raw.info, meg=False, eeg=True, stim=False, eog=True,
                       exclude='bads')
    epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=baseline, preload=True, verbose=False)

    #TODO: check why this doesnt always work!!
    epochs.drop_bad(reject=dict(eog=200e-5), flat=None)
    evoked = epochs['66'].average()
    evoked.plot_topomap(times=np.array([0, 0.050, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), time_unit='s')
    evoked.plot();


#%% SHOW EVENTS
event_id = list(set(events[:,2])) #[3, 13, 66]
epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=True, verbose=False)
mne.viz.plot_events(events, data_raw.info['sfreq'], data_raw.first_samp, event_id=epochs.event_id)


#%% SHOW EPOCHS (e.g. for only one type of stimuli)
#events_select = mne.pick_events(events, include=[13])
#mne.viz.plot_events(events_select)
epochs['66'].plot(events=events_select)

# The plot is interactive, so work with it selec, zoom etc. 
# Then save if wanted with:
# plt.savefig('test.pdf')


#%% SHOW SIGNALS AND STIMULI
order = np.arange(data_raw.info['nchan'])
order[9] = 64  # We exchange the plotting order of two channels
#order[0] = 30
#order[1] = 31
#order[30] = 0
#order[31] = 1
order[64] = 9  # to show the trigger channel as the 10th channel.
#data_raw.plot(n_channels=10, order=order, block=True)
#%%
data_raw.plot(events=events, n_channels=10, order=order)

#%%
layout = mne.channels.read_layout('Vectorview-mag')
layout.plot()
data_raw.plot_sensors(kind='topomap', title = file, show_names = True)
#data_raw.plot_psd_topo(tmax=30., fmin=0, fmax=200, n_fft=1024, layout=layout)

#%% EOG artefact correction
eog_events = mne.preprocessing.find_eog_events(data_raw)
n_blinks = len(eog_events)
# Center to cover the whole blink with full duration of 0.5s:
onset = eog_events[:, 0] / data_raw.info['sfreq'] - 0.25
duration = np.repeat(0.5, n_blinks)
annot = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                        orig_time=data_raw.info['meas_date'])
data_raw.set_annotations(annot)
print(data_raw.annotations)  # to get information about what annotations we have
data_raw.plot(events=eog_events)  # To see the annotated segments.

#%% EOG artefact correction 2
reject = dict(eog=150e-5)

events = mne.find_events(data_raw, shortest_event=0, stim_channel='STI 014', verbose=False)
event_id = [3, 13, 66]
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.8  # end of each epoch (500ms after the trigger)
baseline = (None, 0)  # means from the first instant to t = 0
picks_meg = mne.pick_types(data_raw.info, meg=False, eeg=True, eog=True,
                           stim=False, exclude='bads')
epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks_meg, baseline=baseline, reject=reject,
                    reject_by_annotation=True)

#%% EOG artefact correction --- TRY OWN SOLUTION:
heog_dist = data[30,:]
fig = plt.subplots(figsize=(10,8))
plt.hist(heog_dist)




#%% ICA artifact correction (from mne)
from mne.preprocessing import ICA

#Import data
file = PATH_DATA + cnt_files[2]
data_raw = mne.io.read_raw_cnt(file, montage=None, eog='auto', preload=True)
    
# Band-pass filter: 
data_raw.filter(1, 40, fir_design='firwin')

suspects, suspects_names = helper_functions.select_bad_channels(data_raw, 60, threshold=5)
data_raw.info['bads'] = suspects_names

method = 'fastica'
#method = 'extended-infomax'

# Choose other parameters
n_components = 25  # if float, select n_components by explained variance of PCA
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23

ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)

picks_meg = mne.pick_types(data_raw.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')

#projs, data_raw.info['projs'] = data_raw.info['projs'], []
ica.fit(data_raw, picks=picks_meg, decim=decim)
#data_raw.info['projs'] = projs
print(ica)


#%%
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

eog_average = create_eog_epochs(data_raw, picks=picks_meg).average()

eog_epochs = create_eog_epochs(data_raw)  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).

ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course


#%%
ica.apply(data_raw)
data_raw.plot()  # check the result

#%%
event_id = list(set(events[:,2])) #[3, 13, 66]
epochs = mne.Epochs(data_raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=True, verbose=False)
epochs['66'].plot(events=events_select)


#%%
evoked = epochs['66'].average()
evoked.plot(spatial_colors=True);

#%%
evoked.plot_topomap(times=np.array([0, 0.050, 0.1, 0.2, 0.4, 0.6, 0.8]), time_unit='s')

#%%
projs, events = mne.preprocessing.compute_proj_eog(data_raw, n_grad=2, n_mag=2, n_eeg=2, average=True)
print(projs)

eog_projs = projs[-3:]
mne.viz.plot_projs_topomap(eog_projs, info=data_raw.info)