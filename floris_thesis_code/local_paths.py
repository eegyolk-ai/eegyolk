import os

# STORAGE
storage = "/volume-ceph"

# Location for saving trained models:
# Pretrained models are stored in the eegyolk repository
user = os.path.expanduser("~")
models = os.path.join(user, "eegyolk", "floris_thesis_code", 
                      "models", "trained_models")

# ePodium project
ePod = os.path.join(storage, "ePodium_projectfolder")
ePod_dataset = os.path.join(ePod, "dataset")
ePod_dataset_events = os.path.join(ePod, "events")
ePod_metadata = os.path.join(ePod, "metadata")
ePod_epochs = os.path.join(ePod, "epochs_fif")
ePod_epochs_ddp_dims = os.path.join(ePod, "epochs_fif_500Hz_26ch") 
ePod_epochs_events = os.path.join(ePod_epochs, "events")

# DDP
DDP = os.path.join(storage, "DDP_projectfolder")
DDP_dataset = os.path.join(DDP, "dataset")
DDP_metadata = os.path.join(DDP, "metadata")
DDP_epochs = os.path.join(DDP, "epochs_fif")
DDP_epochs_events = os.path.join(DDP_epochs, "events")

