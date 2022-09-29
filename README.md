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
    run following line on your machine: `conda env create -f current_enviro2.yml` and switch to this environment running command: `conda activate envyolk`.
2. Update the configuration_template.py (NOT config_template) file and rename to config.py.
3. (being rebuilt) Use the preprocessing notebooks to process the raw data to usable data for either the ML or (reduced) DL models (separate notebooks).
4. (being rebuilt) The 'model training' notebooks can be used the train and save models.
5. (being rebuilt) The 'model validation' notebooks can be used to assess the performance of the models.

## Testing
Testing uses synthetic data. Testing will requires you to either run tests inside a container or extract the data from our image with synthetic data in our docker. The docker image will be `drcandacemakedamoore/eegyolk-test-data:latest` . Until then you could also reconfigure and rename your own valid bdf files and metadata as configured and named in the tests/test.py, and local testing should work. 
Finally, you can contact Dr. Moore c.moore@esciencecenter.nl for synthetic test data and/or with any questions on testing.


# Installing

This has only been tested on Linux so far.

    python -m venv .venv
    . .venv/bin/activate
    ./setup.py install


# Configuring

In order to preprocess and to train the models the code needs to be
able to locate the raw data and the metadata, and for the training
it also needs the preprocessed data to be available.

There are several ways to specify the location of the following
directories:

-   **root:** Special directory.  The rest of the directory layout can
    be derived from its location.
-   **data:** The location of raw CNT data files.  This is the directory
    containing `11mnd mmn` and similar files.
-   **metadata:** The location of metadata files.  This is the directory
    that contains `ages` directory, which, in turn, contains files
    like `ages_11mnths.txt`.
-   **preprocessed:** The directory that will be used by preprocessing
    code to output CSVs and h5 files.  This directory will be used
    by the model training code to read the training data.
-   **models:** The directory to output trained models to.

You can store this information persistently in several locations.

1.  In the same directory where you run the script (or the notebook).
    Eg. `./config.json`.
2.  In home directory, eg. `~/.eegyolk/config.json`.
3.  In global directory, eg `/etc/eegyolk/config.json`.

This file can have this or similar contents:

    {
        "root": "/mnt/data",
        "metadata": "/mnt/data/meta",
        "preprocessed": "/mnt/data/processed"
    }

The file is read as follows: if the files specifies `root`
directory, then the missing entires are assumed to be relative to
the root.  You don't need to specify the root entry, if you specify
all other entires.


# Command-Line Interface

You can preprocess and tain the models using command-line interface.

Below are some examples of how to do that:

This will pre-process the first ten CNT files in the
`/mnt/data/original-cnts` directory.

    python -m eegyolk acquire \
           --input /mnt/data/original-cnts \
           --metadata /mnt/data/metadata \
           --output /mnt/data/preprocessed \
           --limit 10

This will train a model using dummy algorithm.  In case of dummy
algorithm both `best_fit` and `fit` do the same thing.

    python -m eegyolk ml \
           --input /mnt/data/preprocessed \
           --output /mnt/data/trained_models \
           --size 100 \
           dummy best_fit

Similarly, for neural networks training and assessment:

    python -m eegyolk nn \
           --input /mnt/data/preprocessed \
           --output /mnt/data/trained_models \
           --epochs 100 \
           train_model 1

It's possible to load configuration (used for directory layout) from
alternative file:

    python -m eegyolk --config /another/config.json ml \
           --input /mnt/data/preprocessed \
           --output /mnt/data/trained_models \
           --size 100 \
           dummy best_fit

All long options have short aliases.
