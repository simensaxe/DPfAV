import os
import copy
import numpy as np
import pickle as pkl
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import transform_matrix
from nuscenes.prediction import PredictHelper



class CustomNuScenes():
    def __init__(self, data_root, version):
        # initalize nuscenes object
        if os.path.exists(f'nusc_{version}.pkl'):
            print('File exists')
            with open(f'nusc_{version}.pkl', 'rb') as f:
                nusc = pkl.load(f)
        else:
            print('File does not exist, loading from scratch')
            nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
            print("Saving to file")
            with open(f'nusc_{version}.pkl', 'wb') as f:
                pkl.dump(nusc, f)
        
        self.nusc = nusc
        self.data_root = data_root
        self.pred_helper = PredictHelper(self.nusc)
    
    def get_tokens_in_scene(self, scene):
        sample_token = scene['first_sample_token']
        sample_list = []
        while sample_token != '':
            sample_list.append(sample_token)
            sample_token = self.nusc.get('sample', sample_token)['next']
        return sample_list
    
    def get_n_next_tokens(self, token, n):
        sample_token = token
        sample_list = []
        for i in range(n):
            if sample_token == '':
                last_token = sample_list[-1]
                sample_list.append(last_token)
                continue
            sample_list.append(sample_token)
            sample_token = self.nusc.get('sample', sample_token)['next']
        return sample_list
    
    def get_n_prev_tokens(self, token, n):
        sample_token = token
        sample_list = []
        for i in range(n):
            sample_token = self.nusc.get('sample', sample_token)['prev']
            if sample_token == '':
                last_token = sample_list[-1]
                sample_list.append(last_token)
                continue
            sample_list.append(sample_token)
        # reverse list
        sample_list.reverse()
        return sample_list

    def get_front_cam_filepath(self, token):
        sample = self.nusc.get('sample', token)
        front_cam_token = sample['data']['CAM_FRONT']
        front_cam_file = self.nusc.get('sample_data', front_cam_token)
        front_cam_filepath = os.path.join(self.data_root, front_cam_file['filename'])
        return front_cam_filepath

    def get_ego_pose(self, token, sensor='CAM_FRONT'):
        '''
        returns the ego_pose for the sensor
        '''
        sample = self.nusc.get('sample', token)
        front_cam_token = sample['data'][sensor]
        front_cam_file = self.nusc.get('sample_data', front_cam_token)
        ego_pose_token = front_cam_file['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        return ego_pose
        
    
    def transform_ego_poses(self, ego_poses, origin_idx):
        '''
        Sets the origin_idx ego pose to be the origin of the world frame.
        param: ego_poses: nuscenes dict of ego poses
        return: ego_poses_list: list of only the ego poses
        return: yaw_list: list of yaw angles
        return: state_vector_list: list of state vectors (x, y, yaw)
        '''
        ego_poses_list = []
        yaw_list = []
        state_vector_list = []
        origin_ego_pose = ego_poses[origin_idx]
        translation = origin_ego_pose['translation']
        rotation_quat = origin_ego_pose['rotation']
        origin_yaw = Quaternion(rotation_quat).yaw_pitch_roll[0]
        # set transformation to origin
        T_inv = transform_matrix(np.array(translation), Quaternion(rotation_quat), inverse=True)
        T_s_g = transform_matrix(np.array(translation), Quaternion(rotation_quat), inverse=False)
        for ego_pose in ego_poses:
            # get positions
            translation = copy.deepcopy(ego_pose['translation'])
            translation.append(1)
            x_new = T_inv @ translation
            ego_poses_list.append(x_new)
            # get yaw
            rotation_quat = Quaternion(copy.deepcopy(ego_pose['rotation'])).yaw_pitch_roll[0]
            yaw = rotation_quat - origin_yaw
            yaw_list.append(yaw)
            # get state vector
            state_vector = [x_new[0], x_new[1], yaw]
            state_vector_list.append(state_vector)
    
        return ego_poses_list, yaw_list, state_vector_list, T_inv, T_s_g, origin_yaw
    
    
    def get_front_cam_intrinsics(self, first_token):
        '''
        Returns the intrinsic matrix of the front camera
        '''
        calibrated_cam_front = self.nusc.get('calibrated_sensor',
                                              self.nusc.get('sample_data',
                                                            self.nusc.get('sample',
                                                                        first_token)['data']['CAM_FRONT'])['calibrated_sensor_token'])
        intrinsic = calibrated_cam_front['camera_intrinsic']
        return intrinsic
    
    def get_T_body_front_cam(self, first_token):
        '''
        Returns the T matrix for camera to body
        '''
        calibrated_cam_front = self.nusc.get('calibrated_sensor',
                                              self.nusc.get('sample_data',
                                                             self.nusc.get('sample',
                                                                            first_token)['data']['CAM_FRONT'])['calibrated_sensor_token'])
        T_front_cam_body = transform_matrix(calibrated_cam_front['translation'], Quaternion(calibrated_cam_front['rotation']), inverse=True)
        return T_front_cam_body
    
    def get_vel_and_steering_angle(self, vehicle_data):
        '''
        Returns the steering angle and velocity of the vehicle
        '''
        steering_wheel_angle = np.deg2rad(vehicle_data['steering'])
        steering_ratio = (2.73*np.pi)/(0.552) # from nuscenes docs
        steering_angle = steering_wheel_angle / steering_ratio
        velocity = vehicle_data['vehicle_speed']/3.6 # convert to m/s
        return [velocity, steering_angle]
    
    def token_index(self, nbr_keyframes, token):
        idx = 0
        while token != '':
            token = self.nusc.get('sample', token)['next']
            idx += 1
        
        return nbr_keyframes - idx
        
