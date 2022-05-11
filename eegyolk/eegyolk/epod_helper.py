import numpy as np      # numerical computing (manipulating and performing operations on arrays of data)
import copy             # Can Copy and Deepcopy files so original file is untouched

def group_event_markers(event_markers):
    '''
    Specific to the ePODIUM dataset.
    Reduces the number of distinctive events from 78 to 12 events.
    This is done by combining different pronounciations into the same event.
    '''    
    event_markers_12 = copy.deepcopy(event_markers)
    for i in range(len(event_markers)):
        for newValue, minOld, maxOld in event_conversion_12:
            condition = np.logical_and(minOld <= event_markers_12[i], event_markers_12[i] <= maxOld)
            event_markers_12[i] = np.where(condition, newValue, event_markers_12[i])
    return event_markers_12

event_dictionary = {
    'GiepMT_FS': 1,
    'GiepMT_S': 2,
    'GiepMT_D': 3,
    'GiepST_FS': 4,
    'GiepST_S': 5,
    'GiepST_D': 6,
    'GopMT_FS': 7,
    'GopMT_S': 8,
    'GopMT_D': 9,
    'GopST_FS': 10,
    'GopST_S': 11,
    'GopST_D': 12,
}

event_conversion_12 = [
    [1, 1, 12], 
    [2, 13, 24], 
    [3, 25, 36], 
    [4, 101, 101], 
    [5, 102, 102], 
    [6, 103, 103],
    [7, 37, 48],
    [8, 49, 60],
    [9, 61, 72],
    [10, 104, 104],
    [11, 105, 105], 
    [12, 106, 106]
]
