import h5py
from pathlib import Path
from pynwb import read_nwb
from NWB_reader_functions import *

def filter_ephys_nwb_files(nwb_dir):
    matched = []

    for fpath in Path(nwb_dir).glob("*.nwb"):

        try:
            # Open only the HDF5 structure — extremely fast
            with h5py.File(fpath, "r") as f:

                # Check that the NWB Units table exists and is nonempty
                if "units" not in f:
                    continue

                units_group = f["units"]

                # NWB Units is a DynamicTable
                # It contains a `id` dataset with N entries
                if "id" not in units_group:
                    continue

                n_units = len(units_group["id"])
                if n_units == 0:
                    continue

                # Passed all filters
                matched.append(fpath)

        except Exception:
            # File corrupted / unreadable → skip
            continue

    return matched
