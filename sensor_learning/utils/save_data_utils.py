#!/usr/bin/env python3
import h5py as h5
import hdf5view as h5view
import numpy as np
import os

import logging
import sensor_learning.utils.logging_conf
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_trial_data(trial_data: dict) -> str:
    """
    Save the given pose, color image, and depth image to an HDF5 file.
    
    trial_000.h5
    ├── timesteps       # (15,) - sim step index or actual time in seconds
    ├── poses/
    │   ├── positions       # (15, 3)
    │   └── orientations    # (15, 4)
    └── sensors/
        └── <sensor_name>/
            ├── extrinsics  # (15, 4, 4)
            └── depth       # (15,)

    :param trial_data: A dictionary containing the trial data to be saved.
    :return: The file path of the saved HDF5 file.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(pkg_dir, "data")
    file_path = os.path.join(data_dir, trial_data['trial_name'] + ".h5")
    if not os.path.exists(data_dir):
        logger.info(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir)
    if os.path.exists(file_path):
        logger.warning(f"File {file_path} already exists. It will be overwritten.")
        
    with h5.File(file_path, "w") as file:
        file.attrs["trial_name"] = trial_data['trial_name']
        file.attrs["tree_id"] = trial_data['tree_id']
        file.attrs["tree_type"] = trial_data['tree_type']
        
        # Generated poses
        generated_poses_group = file.create_group('generated_poses')
        generated_poses_group.attrs['x_range'] = trial_data['eef_poses']['generated']['x_range']
        generated_poses_group.attrs['y_range'] = trial_data['eef_poses']['generated']['y_range']
        generated_poses_group.attrs['z_range'] = trial_data['eef_poses']['generated']['z_range']
        generated_poses_group.attrs['theta_range'] = trial_data['eef_poses']['generated']['theta_range']
        generated_poses_group.attrs['phi_range'] = trial_data['eef_poses']['generated']['phi_range']
        generated_poses_group.attrs['points_per_axis'] = trial_data['eef_poses']['generated']['points_per_axis']
        generated_poses_group.attrs['angles_per_axis'] = trial_data['eef_poses']['generated']['angles_per_axis']
        generated_poses_group.attrs['start_orientation'] = trial_data['eef_poses']['generated']['start_orientation']
        generated_poses_group.attrs['start_position'] = trial_data['eef_poses']['generated']['start_position']
        generated_poses_group.create_dataset('poses', data=trial_data['eef_poses']['generated']['poses'])
        
        # Actual poses (from PyBullet sim)
        actual_poses_group = file.create_group('actual_poses')
        actual_poses_group.create_dataset('positions', data=trial_data['eef_poses']['actual']['position'])
        actual_poses_group.create_dataset('orientations', data=trial_data['eef_poses']['actual']['orientation'])
        
        simulation_steps_group = file.create_group('simulation_steps')
        simulation_steps_group.attrs['time_step_interval'] = trial_data['time_step_interval']
        simulation_steps_group.create_dataset('steps', data=trial_data['sim_steps']) # default timestep is 1/240
        
        sensors_group = file.create_group('sensors')
        for sensor_name, sensor_data in trial_data['sensors'].items():
            sensor_group = sensors_group.create_group(sensor_name)
            sensor_group.create_dataset('extrinsics', data=sensor_data['extrinsics'])            
            for mode, mode_data in sensor_data['mode'].items():
                mode_group = sensor_group.create_group(mode)
                if mode_data['rgb'] is not None:
                    mode_group.create_dataset('rgb', data=mode_data['rgb'])
                if mode_data['depth'] is not None:
                    mode_group.create_dataset('depth', data=mode_data['depth'])
                if mode_data['pointcloud'] is not None:
                    mode_group.create_dataset('pointcloud', data=mode_data['pointcloud'])

    return file_path