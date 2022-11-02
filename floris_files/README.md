# Predicting age and dyslexia with deep learning on ERPs

This codebase is part of the *eegyolk* repository. The notebooks and functions are created for the master thesis ... by Floris Pauwels. The goal of this code is to predict age and dyslexia from EEG and ERP data. 

#### ePodium project
The thesis is part of the project *ePODIUM: early Prediction Of Dyslexia in Infants Using Machine learning*. The project is a collaboration between researchers from Utrecht University, UMC Utrecht, and the eScience Center in Amsterdam. The goal of the ePodium project is to explore if EEG data measured in infancy can predict the occurrence of later literacy difficulties in individual children. 



#### Previous results
As a part of this project, the master student B.M.A. Bruns has already been successful at predicting the age of children between 11 and 47 months from EEG signals, by applying deep learning models to the *Dutch Dyslexia Program* (DDP) dataset. The code for this project is available at https://github.com/epodium/EEG_age_prediction.




## Files

### Notebooks

This codebase contains five notebooks. Some of these notebooks contain widgets to interact with the notebooks and modify the data:

* `epodium.ipynb` uses the ePodium dataset. The data is loaded and the experiment is explained and visualised. The data is processed and sorted into epochs. Finally ERPs are plotted from these epochs.
* `ddp.ipynb` is similar to the 'ePodium' notebook, but now the *Dutch Dyslexia Project* dataset is used.

* `model_training.ipynb` trains deep neural network models to predict age and risk of dyslexia on the datasets. Different models and hyperparameters can be chosen. The trained model with the lowest loss on the validation set is stored for later analysis.
* `model_analysis.ipynb` analyses the input data and models trained in the 'model_training' notebook. The model accuracies on the test set are calculated and the predictions are plotted against the actual values.

* `simulated_data.ipynb` is a demo file to create and visualise simulated EEG data from the signal frequencies.

### Functions
* `epodium.py` contains the epodium class for working with the ePodium dataset. The class contains data and methods specific to the dataset.
* `ddp.py`  contains the ddp class for working with the DDP dataset. Like the epodium class, this class contains data and methods specific to the dataset.

* `processing.py` contains tools for (multi)processing raw EEG data and saves the processed EEG data as epochs.
* `sequences.py` contains a sequence class for each dataset to iterate over the data for training the deep learning model.


* `display_helper.py` contains tools to help with displaying information.
* `simulated_data.py` contains functions for creating simulated data.
* `data_io.py` contains functions to save and load data to and from storage.



### Models
The 'models' folder contains the model architectures from two repositories.
+ The models from [Deep Learning for Time Series Classification](https://github.com/hfawaz/dl-4-tsc) by 'hfawaz' are stored in `dl_4_tsc.py`. 
+ The models from [EEG Deep Learning](https://github.com/SuperBruceJia/EEG-DL) by 'SuperBruceJia' are stored in `eeg_dl.py`. 

The model folder also contains the original models from the repositories and a trained_models folder in which the trained models are stored.


## Dependencies
### Dutch Dyslexia Program (DDP) dataset

In the DDP dataset 300 children were followed from the age of 2 months up to 9 years. The EEG-signals were measured from the children every 6 months between 2 and 47 months. In these experiments the EEG-signals were recorded from the children, where the children listen to the dutch words "bak" and "dak". Nine variation of these sounds were played, each a different combination of the two words. The event-related potentials to each distinct sound event was measured. 

A portion of the dataset is available on the [DANS](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:112935/) website. NLeSC employees can download the data from [surfdrive](https://surfdrive.surf.nl/files/index.php/s/mkwBAisnYUaPRhy).

### ePodium dataset
A dataset is developed by the ePodium project to predict the risk of dyslexia in toddlers. In the experiment EEG-data is collected from 129 toddlers between the age of 16 and 24 months. The children perform two half-hour tests in a 3 month interval in which the EEG-data is recorded. 

The test uses the auditory oddball paradigm. The hypothesis is that children with risk of dyslexia are not as skilled at distinguishing spoken syllables as typical children. If this hypothesis is true, dyslexic children can potentially be diagnosed based on an abnormality in the mismatch responses from standard and deviant syllables. The dataset contains information on the reading skills of the parents and a child is at risk of dyslexia if the parents are dyslexic, since there is a correlation between dyslexia among parents and children. Children at risk of dyslexia are estimated to have a less distinct mismatch response than typical children.

Contact Candace Makeda Moore (c.moore@esciencecenter.nl) to ask for access to the data available to NLeSC employees.

### Packages
* [MNE](https://mne.tools/) is the go-to package for exploring, visualizing, and analyzing neurophysiological data such as EEGs. 
* [Tensorflow](https://www.tensorflow.org/) is a deep learning framework for training and inference of deep neural networks. Tensorflow contains the [Keras](https://keras.io/)  library which contains the building blocks of neural networks to make it as easy as possible to create such a network.
* [Autoreject](https://autoreject.github.io/)  is a library to automatically reject bad trials and repair bad sensors in EEG data. This library is used for processing the data.

* [Numpy](https://numpy.org/) is a widely used package for math and arrays.
* [Pandas](https://pandas.pydata.org/) is a package for loading and managing data structures.
* [Matplotlib](https://matplotlib.org/) can visualize data such as graphs and scatterplots.
* [IPyWidgets](https://ipywidgets.readthedocs.io/) enables interactive widgets for Jupyter notebooks. 

* [Scikit-learn](https://scikit-learn.org/) is a package for machine learning and data analysis. In this repository the package is used to split the data into train, test, and validation sets with the *train_test_split* function.
* [Scipy](https://scipy.org) is a package for scientific computation. In this repository the fourier transform is used for simulating data.
* [wave](https://docs.python.org/3/library/wave.html) is an optional module to analyse the WAV sound format. These WAV sounds are played as a stimulus to the subjects.

* [os](https://docs.python.org/3/library/os.html), [glob](https://docs.python.org/3/library/glob.html), [multiprocessing](https://docs.python.org/3/library/multiprocessing.html), and [random](https://docs.python.org/3/library/random.html) are standard packages included in the Python installation. 

### Install guide
To use the notebooks, you can download the files in this repository to your system. The code also needs access to a dataset. Finally the required packages need to be installed in Python on your system. This can be done with pip or conda in Windows or sudo in Linux.

The `local_paths.py` file contains paths to the saved models and the dataset files. Make sure the paths are correctly set in this file.