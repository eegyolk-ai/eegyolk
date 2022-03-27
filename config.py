# config.py template
# Add your respective folder names and save as config.py
# Please note that there are two roots used here, these can be the same root
import os


# PATH_CODE = os.path.join(ROOT, "EEG_explorer")
#
# PATH_OUTPUT = os.path.join(ROOT, "Output_EEG")

# Root folder (EEG_age_prediction/)
ROOT = "C:/Projects"

# Second root for large data storage (e.g. External HDD)
SECOND_ROOT = os.path.join(ROOT, "EEG_explorer", "ePodium")

# Saved models
local_folder = "EEG_age_prediction"
PATH_MODELS = os.path.join(ROOT, local_folder, "trained_models/")

# Processed data for DL models
PATH_DATA_PROCESSED_DL = os.path.join(ROOT, local_folder, "Data/data_processed_DL/")

# Processed data for ML models
PATH_DATA_PROCESSED_ML = os.path.join(ROOT, local_folder, "Data/data_processed_ML/")

# EEG metadata folder
PATH_METADATA = os.path.join(ROOT, "EEG_explorer", "ePODIUM_Metadata")

# Raw EEG data folder
PATH_RAW_DATA = os.path.join(ROOT, "EEG_explorer", "Data_Old")

# # Already preprocessed data for initial experiments - before using own preprocessing pipeline
PATH_DATA_PROCESSED_OLD = os.path.join(SECOND_ROOT, "Preprocessed_old/Data/data_processed_old/")