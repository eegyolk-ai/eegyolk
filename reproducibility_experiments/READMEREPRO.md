<p align="center">
    <img style="width: 30%; height: 30%" src="https://github.com/NLeSC/ePodium/blob/main/eegyolk_logo.png">
</p>

# eegyolk




The notebooks include an updated version of a project for deep learning for age prediciton using EEG data, as well as new ongoing work from students at the University of Urtrecht. 
To reproduce previous work with our libraray you are best off
running on a Linux machine. Some functions will not produce the same result
on another machine type. Below are instructions:

### Notebooks
The reproduction work is currently in notebooks/original/remastered.
Complete the pre-processing on the data, then apply traditional-ml-models
notebook's code. 


### Configuration file





### Program files



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
