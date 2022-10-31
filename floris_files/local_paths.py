import os

# STORAGE
storage = "/volume-ceph"
user = os.path.expanduser("~")

# Location for saving trained models:
# Pretrained models are stored in the eegyolk repository
models = os.path.join(user, "eegyolk", "floris_files", 
                      "models", "trained_models")

# ePodium project
ePod = os.path.join(storage, "ePodium_projectfolder")
ePod_dataset = os.path.join(ePod, "dataset")
ePod_dataset_events = os.path.join(ePod, "events")
ePod_metadata = os.path.join(ePod, "metadata")
ePod_epochs = os.path.join(ePod, "epochs_fif")
ePod_epochs_events = os.path.join(ePod_epochs, "events")

# DDP
DDP = os.path.join(storage, "DDP_projectfolder")
DDP_dataset = os.path.join(DDP, "dataset")
DDP_metadata = os.path.join(DDP, "metadata")
DDP_epochs = os.path.join(DDP, "epochs_fif")
DDP_epochs_events = os.path.join(DDP_epochs, "events")

