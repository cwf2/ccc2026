'''data.py - working with data archived on OSF'''

import os
from osfclient import OSF
from ccc2026 import config

def download(filenames):
    '''download files from the OSF archive'''

    # allow single filename to be passed instead of a list
    if isinstance(filenames, str):
        filenames = [filenames]

    # initiate connection
    osf = OSF()
    project = osf.project(config.OSF_PROJECT)
    storage = project.storage()

    # iterate over storage, pull requested files
    for file in storage.files:
        if file.name in filenames:
            with open(os.path.join(config.DATA_DIR, file.name), "wb") as fh:
                file.write_to(fh)
