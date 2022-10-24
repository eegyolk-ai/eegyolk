import os

# STORAGE
storage = "/volume-ceph"

## Personal storage
# Location for saving trained models:
models = os.path.join("/home", "fpauwels", "eegyolk",
                      "floris_files", "models", "trained_models")
# personal_storage = os.path.join(storage, "floris_storage")
# ePod_personal = os.path.join(personal_storage, "epod")
# DDP_personal = os.path.join(personal_storage, "ddp")

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

