<p align="center">
    <img style="width: 30%; height: 30%" src="https://github.com/NLeSC/ePodium/blob/main/eegyolk_logo.png">
</p>

# eegyolk

[![PyPI](https://img.shields.io/pypi/v/eegyolk.svg)](https://pypi.python.org/pypi/eegyolk/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6865762.svg)](https://doi.org/10.5281/zenodo.6865762)
[![Anaconda-Server Badge](https://anaconda.org/eegyolk/eegyolk/badges/version.svg)](https://anaconda.org/eegyolk/eegyolk)
[![Sanity](https://github.com/eegyolk-ai/eegyolk/actions/workflows/on-commit.yml/badge.svg)](https://github.com/eegyolk-ai/eegyolk/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/eegyolk-ai/eegyolk/actions/workflows/on-tag.yml/badge.svg)](https://github.com/eegyolk-ai/eegyolk/actions/workflows/on-tag.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This library contains functions, scripts and notebooks for machine learning related to EEGs (electroencephalograms). The notebooks include an updated version of a project for deep learning for age prediciton using EEG data, as well as new ongoing work from students at the University of Urtrecht.

### Notebooks

Initial experiments:
- Ongoing


### Configuration file

The config_template.py file should be renamed to config.py. Here the paths of the file locations can be stored. The ROOT folder can be the ROOT folder of this repository as well.

The Data folder contains the following folder/files:




### Program files

The main program in this repository contains functions, for example DataGenerators.


## Data sets

Some of the data sets of this project are publicly available, and some are not  as they contains privacy-sensitive information.

Original published data from the DDP (Dutch Dyslexia Program) is used as demo data wherever possible. This data can be obtained from:
https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:112935/ 

Collection of newer data acquired to detect dyslexia on a different protocol ended in 2022. This data is not yet public, however, there are many public EEG datasets to which the functions in this library can be applied.

NLeSC employees can download some additional data from [surfdrive](https://surfdrive.surf.nl/files/index.php/s/mkwBAisnYUaPRhy).
Contact Candace Makeda Moore (c.moore@esciencecenter.nl) to discuss additional access to data,

## Getting started

How to get the notebooks running? Assuming the raw data set and metadata is available.

1. Install all Python packages required, using conda and the environment-march-update2.yaml file.
    run following line on your machine: `conda env create -f current_enviro2.yml` and switch to this environment running command: `conda activate mne-marchez`.
2. Update the configuration_template.py (NOT config_template) file and rename to config.py.
3. (being rebuilt) Use the preprocessing notebooks to process the raw data to usable data for either the ML or (reduced) DL models (separate notebooks).
4. (being rebuilt) The 'model training' notebooks can be used the train and save models.
5. (being rebuilt) The 'model validation' notebooks can be used to assess the performance of the models.

## Testing
Testing uses synthetic data. At present testing requires you to either run tests inside a container or extract the data from our image with synthetic data in our docker. The docker image is `drcandacemakedamoore/eegyolk-test-data:latest` . You could also reconfigure and rename your own valid bdf files and metadata as configured and named in the tests/test.py, and local testing should work. 
Finally, you can contact Dr. Moore c.moore@esciencecenter.nl with any questions on testing. 