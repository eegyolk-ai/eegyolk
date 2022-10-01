import os

# STORAGE
storage = "/volume-ceph"
personal_storage = os.path.join(storage, "floris_storage")


# Storage data
epod_personal = os.path.join(personal_storage, "epod")
epod_processed = os.path.join(epod_personal, "clean")
epod_models = os.path.join(epod_personal, "models")
epod_split = os.path.join(epod_personal, "split") # Deprecated


ddp_personal = os.path.join(personal_storage, "ddp")
ddp_clean = os.path.join(ddp_personal, "clean")
ddp_models = os.path.join(ddp_personal, "models")


# ePodium project
ePod = os.path.join(storage, "ePodium_projectfolder")
ePod_dataset = os.path.join(ePod, "dataset")
ePod_processed = os.path.join(ePod, "epochs_fif")
ePod_events = os.path.join(ePod, "events")
ePod_metadata = os.path.join(ePod, "metadata")


# DDP
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


