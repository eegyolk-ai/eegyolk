def init():
    import os
    
    # STORAGE
    global storage, DDP   
    global ePod, ePod_dataset, ePod_events, ePod_metadata
    
    storage = "/volume-ceph"
    
    DDP = os.path.join(storage, "DDP_projectfolder")
    
    ePod = os.path.join(storage, "ePodium_projectfolder")
    ePod_dataset = os.path.join(ePod, "dataset")
    ePod_events = os.path.join(ePod, "events")
    ePod_metadata = os.path.join(ePod, "metadata")
    
    # WORKSPACE
    global home, user, repo, code    
    home = "/home"
    user = os.path.join(home, "fpauwels")
    repo = os.path.join(user, "eegyolk")
    code = os.path.join(repo, "floris_files")