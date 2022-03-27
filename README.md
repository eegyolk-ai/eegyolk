# EEG_scripts

The following are scripts and notebooks for machine learning related to EEGS. They include an updated version of a project for deep learning for age prediciton using EEG data, as well as new ongoing work.

## Structure of files


### Notebooks

Initial experiments:
- Ongoing


### Configuration file

The config_template.py file should be renamed to config.py. Here the paths of the file locations can be stored. The ROOT folder can be the ROOT folder of this repository as well.

The Data folder contains the following folder/files:




### Helper files

The main folder of this repository will also contains a few helper files, for example DataGenerators.

### Scripts




## Data sets

Some of the data sets of this project are publicly available, and some are not  as they contains privacy-sensitive information.

Original published data is used as demo data wherever possible. This data can be obtained from:
https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:112935/ 

NLeSC employees can download the data from [surfdrive](https://surfdrive.surf.nl/files/index.php/s/mkwBAisnYUaPRhy).
Contact Pablo Lopez-Tarifa (p.lopez@esciencecenter.nl) for access to the data, 
or Sven van der Burg (s.vanderburg@esciencecenter.nl) 

## Getting started

How to get the notebooks running? Assuming the raw data set and metadata is available.

1. Install all Python packages required, using conda and the environment-march-update2.yaml file.
2. Update the configuration_template.py (NOT config_template) file and rename to config.py.
3. (being rebuilt) Use the preprocessing notebooks to process the raw data to usable data for either the ML or (reduced) DL models (separate notebooks).
4. (being rebuilt) The 'model training' notebooks can be used the train and save models.
5. (being rebuilt) The 'model validation' notebooks can be used to assess the performance of the models.