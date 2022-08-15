import os

# STORAGE

storage = "/volume-ceph"
processed = os.path.join(storage, "processed")
models = os.path.join(storage, "models")

DDP = os.path.join(storage, "DDP_projectfolder")
DDP_5 = os.path.join(DDP, "05mnd mmn")
DDP_11 = os.path.join(DDP, "11mnd mmn")
DDP_17 = os.path.join(DDP, "17mnd mmn")
DDP_23 = os.path.join(DDP, "23mnd mmn")
DDP_29 = os.path.join(DDP, "29mnd mmn")
DDP_35 = os.path.join(DDP, "35mnd mmn")
DDP_41 = os.path.join(DDP, "41mnd mmn")
DDP_47 = os.path.join(DDP, "47mnd mmn")
DDP_dict = {5: DDP_5, 11: DDP_11, 17: DDP_17, 23: DDP_23, 
            29: DDP_29,35: DDP_35, 41: DDP_41, 47: DDP_47}

DDP_DANS = os.path.join(DDP, "DANS")
DDP_metadata = os.path.join(DDP, "metadata")
DDP_processed = os.path.join(processed, "DDP")
DDP_processed_new = os.path.join(processed, "DDP_new")


ePod = os.path.join(storage, "ePodium_projectfolder")
ePod_dataset = os.path.join(ePod, "dataset")
ePod_events = os.path.join(ePod, "events")
ePod_metadata = os.path.join(ePod, "metadata")

ePod_processed_autoreject = os.path.join(processed, "ePod_autoreject")
ePod_processed_autoreject_events = os.path.join(ePod_processed_autoreject, "events")
ePod_processed_autoreject_epochs = os.path.join(ePod_processed_autoreject, "epochs")
ePod_processed_autoreject_epochs_split_downsampled = os.path.join(ePod_processed_autoreject, "epochs_split_downsampled")

# WORKSPACE

home = "/home"
user = os.path.join(home, "fpauwels")
repo = os.path.join(user, "eegyolk")
code = os.path.join(repo, "floris_files")
hashes = os.path.join(code, "other", "hashes")

# repo = "C:\Floris\Python Folder\Thesis Code\eegyolk"
# storage = "D:\EEG Data"
