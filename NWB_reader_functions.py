import pandas as pd
from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries
import ast
import numpy as np
import h5py

"""
This file define NWB reader functions (inspired from CICADA NWB_wrappers).
The goal is that a function is used to extract one specific element from a NWB file to pass it to any analysis
"""

def read_nwb_file(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    #except TypeError:
    #    nwb_data = h5py.File(nwb_file, "r")
    #    #nwb_keys = nwb_file.keys()
    #    #nwb_file.close()
    #    nwb_general = nwb_data['general']
    #    nwb_proc = nwb_data['processing']
    return nwb_data

def get_mouse_id(nwb_file):
    #io = NWBHDF5IO(nwb_file, 'r')
    #nwb_data = io.read()
    nwb_data = read_nwb_file(nwb_file)
    mouse_id = nwb_data.subject.subject_id

    return mouse_id


def get_session_id(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    session_id = nwb_data.session_id

    return session_id

def get_nwb_file_metadata(nwb_file):
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    session_metadata = nwb_data.subject

    return session_metadata

def get_session_metadata(nwb_file):
    """Get session-level metadata.
     Converts string of dictionary into a dictionary."""
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    session_metadata = ast.literal_eval(nwb_data.experiment_description)

    return session_metadata

def get_video_sampling_rate(nwb_file):
    """
    This function extracts the video sampling rate from a NWB file.
    :param nwb_file:
    :return:
    """
    sess_metadata = get_session_metadata(nwb_file)
    cam_freq = int(sess_metadata['camera_freq']) # get camera sampling frequency
    return cam_freq

def get_mouse_relative_weight(nwb_file):
    sess_metadata = get_session_metadata(nwb_file)
    nwb_metadata = get_nwb_file_metadata(nwb_file)
    mouse_session_weight = nwb_metadata.weight
    mouse_reference_weight = sess_metadata['reference_weight']
    if mouse_session_weight == 'na' or mouse_reference_weight == 'na':
        return np.nan
    if mouse_session_weight == 'nan' or mouse_reference_weight == 'nan':
        return np.nan
    if mouse_session_weight == 'Nan' or mouse_reference_weight == 'Nan':
        return np.nan
    if mouse_session_weight == 'NaN' or mouse_reference_weight == 'NaN':
        return np.nan
    if mouse_session_weight == None or mouse_reference_weight == None:
        return np.nan

    rel_weigh_perc = 100 * float(mouse_session_weight) / float(mouse_reference_weight)
    return rel_weigh_perc

def get_dlc_data_dict(nwb_file):
    """
    This function extracts the DLC data from a NWB file.
    Some additional processing is done.
    :param nwb_file:
    :return:
    """
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()

    # Check if it has DLC data
    try:
        module_keys = nwb_data.processing['behavior']['BehavioralTimeSeries'].time_series.keys()
    except KeyError as err:
        print(f"KeyError: {err}. No DLC data found in NWB file: {nwb_file}")
        return None

    bparts = ['jaw_angle', 'jaw_distance', 'jaw_likelihood', 'jaw_velocity',
              'nose_angle', 'nose_distance', 'nose_tip_likelihood',  'nose_velocity',
              'pupil_area', 'pupil_likelihood', 'pupil_area_velocity',
              'tongue_angle', 'tongue_distance', 'tongue_likelihood', 'tongue_velocity',
              'top_nose_distance', 'top_nose_tip_likelihood',  'top_nose_velocity',
              'top_particle_likelihood', 'whisker_angle', 'whisker_velocity']

    dlc_data_dict = {}
    dlc_ts_dict = {}

    # Iterate and convert all bodyparts
    for bpart in bparts:
        try:
            bpart_data = np.array(nwb_data.processing['behavior']['BehavioralTimeSeries'].time_series[bpart].data)
            bpart_ts = np.array(nwb_data.processing['behavior']['BehavioralTimeSeries'].time_series[bpart].timestamps)
        except KeyError as err:
            print(f"KeyError: {err}. {bpart} not found in NWB file (no DLC data). Skipping")
            continue

        if 'likelihood' in bpart or bpart in ['whisker_angle', 'whisker_velocity']:
            conversion = 1
        else:
            conversion = nwb_data.processing['behavior']['BehavioralTimeSeries'].time_series[bpart].conversion

        dlc_data_dict[bpart] = bpart_data*conversion if 'pupil_area' in bpart else bpart_data*(conversion**2)
        dlc_ts_dict[bpart] = bpart_ts

    # ------------------------------
    # Process directly some features
    # ------------------------------

    # Combine nose and top nose to have the norm
    try:
        if dlc_data_dict['top_nose_distance'].shape == dlc_data_dict['nose_distance'].shape:

            dlc_data_dict['nose_norm_distance'] = np.sqrt(dlc_data_dict['top_nose_distance']**2 + dlc_data_dict['nose_distance']**2)
            dlc_data_dict['nose_norm_velocity'] = np.sqrt(dlc_data_dict['top_nose_velocity']**2 + dlc_data_dict['nose_velocity']**2)
            dlc_ts_dict['nose_norm_distance'] = dlc_ts_dict['top_nose_distance']
            dlc_ts_dict['nose_norm_velocity'] = dlc_ts_dict['top_nose_velocity']
        else:
            print(f"Shape mismatch between top_nose and nose distances. Skipping norm computation: {nwb_file}")
            dlc_data_dict['nose_norm_distance'] = dlc_data_dict['nose_distance']
            dlc_data_dict['nose_norm_velocity'] = dlc_data_dict['nose_velocity']
            dlc_ts_dict['nose_norm_distance'] = dlc_ts_dict['nose_distance']
            dlc_ts_dict['nose_norm_velocity'] = dlc_ts_dict['nose_velocity']

    except KeyError as err:
        print(f"KeyError: {err}. {bpart} not found in NWB file (no DLC data). Replacing with NaNs: {nwb_file}")
        dlc_data_dict['nose_norm_distance'] = np.nan
        dlc_data_dict['nose_norm_velocity'] = np.nan
        dlc_ts_dict['nose_norm_distance'] = np.nan
        dlc_ts_dict['nose_norm_velocity'] = np.nan

    # Filter and remove outliers of pupil area using the percentiles
    try:
        dlc_data_dict['pupil_area'] = np.clip(dlc_data_dict['pupil_area'], 0, np.percentile(dlc_data_dict['pupil_area'], 99))
    except KeyError as err:
        print(f"KeyError: {err}. Pupil area not found in NWB file (no DLC data). Replacing with NaNs: {nwb_file}")
        dlc_data_dict['pupil_area'] = np.nan
        dlc_ts_dict['pupil_area'] = np.nan


    # Format as a directory for each fields there is data and timestamp as key
    dlc_data = {}
    for key in dlc_data_dict.keys():
        dlc_data[key] = {'data': dlc_data_dict[key], 'timestamps': dlc_ts_dict[key]}

    return dlc_data


def get_bhv_type_and_training_day_index(nwb_file):
    """
    This function extracts the behavior type and training day index, relative to whisker training start, from a NWB file.
    :param nwb_file:
    :return:
    """
    #io = NWBHDF5IO(nwb_file, 'r')
    #nwb_data = io.read()
    nwb_data = read_nwb_file(nwb_file)

    # Read behaviour_type and day from session_description, encoded at creation as behavior_type_<day>
    description = nwb_data.session_description.split('_')
    if description[0] == 'free':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    elif description[1] == 'psy':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    elif description[1] == 'on':
        behavior_type = description[0] + '_' + description[1] + '_' + description[2]
        day = int(description[-1])
    elif description[1] == 'off':
        behavior_type = description[0] + '_' + description[1] + '_' + description[2]
        day = int(description[-1])
    elif description[1] == 'context':
        behavior_type = description[0] + '_' + description[1]
        day = int(description[2])
    else:
        behavior_type = description[0]
        day = int(description[1])

    return behavior_type, day


def get_trial_table(nwb_file):
    """
    This function extracts the trial table from a NWB file.
    :param nwb_file:
    :return:
    """
    io = NWBHDF5IO(nwb_file, 'r')
    try:
        nwb_data = io.read()
    except TypeError as err:
        print(nwb_file, err)
    nwb_objects = nwb_data.objects
    objects_list = [data for key, data in nwb_objects.items()]
    data_to_take = None

    # Iterate over NWB objects but keep "trial"
    for ind, obj in enumerate(objects_list):
        if 'trial' in obj.name:
            data = obj
            if isinstance(data, TimeSeries):
                continue
            else:
                data_to_take = data
                break
        else:
            continue
    trial_data_frame = data_to_take.to_dataframe()
    return trial_data_frame

def get_behavioral_events(nwb_file):
    """
    This function extracts the behavioral events from a NWB file.
    :param nwb_file:
    :return:
    """

    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    sess_metadata = get_session_metadata(nwb_file)
    cam_freq = int(sess_metadata['camera_freq']) # get camera sampling frequency

    event_keys = nwb_data.processing['behavior']['BehavioralEvents'].time_series.keys()
    beh_event_dict = {}
    for key in event_keys:
        event_ts = np.array(nwb_data.processing['behavior']['BehavioralEvents'].time_series[key].timestamps)

        # Convert video-related events from DeepLabCut to seconds
        if 'dlc' in key:##
            event_ts = event_ts / cam_freq

        beh_event_dict[key] = event_ts

    return beh_event_dict

def get_behavioral_timeseries(nwb_file):
    """
    This function extracts the behavioral timeseries from a NWB file.
    :param nwb_file:
    :return:
    """

    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    event_keys = nwb_data.processing['behavior']['BehavioralTimeSeries'].time_series.keys()
    beh_ts_dict = {}
    for key in event_keys:
        event_ts = nwb_data.processing['behavior']['BehavioralTimeSeries'].time_series[key].timestamps
        beh_ts_dict[key] = np.array(event_ts)

    return beh_ts_dict

def get_electrode_group_table(nwb_file):
    """
    This function extracts the electrode group table from a NWB file.
    :param nwb_file:
    :return:
    """

    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    groups = [e for e in nwb_data.electrode_groups]
    groups_df = pd.DataFrame(data=groups, columns=['electrode_group'])
    return groups_df

def get_electrode_table(nwb_file):
    """
    This function extracts the electrode table from a NWB file.
    :param nwb_file:
    :return:
    """
    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    electrode_table = nwb_data.electrodes.to_dataframe()
    return electrode_table

def get_device_table(nwb_file):
    """
    This function extracts the device table from a NWB file.
    :param nwb_file:
    :return:
    """

    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    devices_group = list(nwb_data.devices)
    device_table = pd.DataFrame(data=devices_group, columns=['electrode_group'])
    return device_table

def get_unit_table(nwb_file):
    """
    This function extracts the unit table from a NWB file.
    :param nwb_file:
    :return:
    """

    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    try:
        unit_table = nwb_data.units.to_dataframe()
        # Create a mouse-specific neuronal id for each cluster_id
        unit_table['neuron_id'] = unit_table.index
        return unit_table
    except AttributeError as e:
        beh, day = get_bhv_type_and_training_day_index(nwb_file)
        if beh=='whisker' and day==0:
            print(f"Unit table in day 0 not found in {nwb_file}")
        return None

def get_unit_spike_times(nwb_file):
    """
    This function extracts the unit spike times from a NWB file.
    :param nwb_file:
    :return:
    """

    io = NWBHDF5IO(nwb_file, 'r')
    nwb_data = io.read()
    unit_spike_times = nwb_data.units.spike_times

    return unit_spike_times




