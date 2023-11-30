import torch
from torch.utils.data import Dataset
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from helpers.custom_nuscenes import CustomNuScenes
from helpers.utils import image_transform, vel_steering_normalizer, position_normalizer



class VelSteeringDataset(Dataset):
    def __init__(self, dataset_path, version, obs_horizon, pred_horizon, action_horizon,
                 test=False, test_tokens_file=None):
        self.nusc = CustomNuScenes(dataset_path, version)
        self.nusc_can = NuScenesCanBus(dataroot=dataset_path)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.test = test
        # need to implement test scheme
        if self.test:
            # use tokens in the test set
            # when sampling, read images with tokens from the test set
            with open(test_tokens_file, 'r') as f:
                self.start_tokens = f.readlines()
                self.start_tokens = [token.strip() for token in self.start_tokens]
            # get scene names for start tokens
            self.scenes = []
            for token in self.start_tokens:
                scene_token = self.nusc.nusc.get('sample', token)['scene_token']
                self.scenes.append(self.nusc.nusc.get('scene', scene_token))
            
            for scene in self.scenes:
                # check if scene has can bus data
                try:
                    vehicle_data = self.nusc_can.get_messages(scene['name'],'vehicle_monitor')
                    if len(vehicle_data) < 35:
                        # to much missing data to pad, throw away scene
                        raise Exception(f'To much missing CAN data, {len(vehicle_data)}')
                except Exception as e:
                    # can bus data not available, drop scene
                    print(e)
                    # pop tokens in scene from start tokens
                    tokens_in_scene = self.nusc.get_tokens_in_scene(scene)
                    for token in tokens_in_scene:
                        self.start_tokens.remove(token)
                    continue
 
        else:    
            # use tokens in the training set
            # when sampling, read images with tokens from the training set
            self.scenes = self.nusc.nusc.scene

            # define possible start tokens that can be sampled from
            self.start_tokens = []
            for scene in self.scenes:
                # check if scene has can bus data
                try:
                    vehicle_data = self.nusc_can.get_messages(scene['name'],'vehicle_monitor')
                    if len(vehicle_data) < 35:
                        # to much missing data to pad, throw away scene
                        raise Exception(f'To much missing CAN data, {len(vehicle_data)}')
                except Exception as e:
                    # can bus data not available, drop scene
                    print(e)
                    continue
                scene_tokens = self.nusc.get_tokens_in_scene(scene)
                for token in scene_tokens[:-pred_horizon]:
                    self.start_tokens.append(token)

    def __len__(self):
        return len(self.start_tokens)
        
    def __getitem__(self, idx):
        # define tokens in sequence from start token to start token + pred_horizon
        inital_frame_token = self.start_tokens[idx]
        tokens_in_sequence = self.nusc.get_n_next_tokens(inital_frame_token,
                                                            self.pred_horizon)
        if len(tokens_in_sequence) != self.pred_horizon:
            print('ERROR', len(tokens_in_sequence), ' ', self.pred_horizon, ' ', idx)
        
        # get images and ego_poses
        image_paths = []
        ego_poses = []
        for token in tokens_in_sequence:
            image_paths.append(self.nusc.get_front_cam_filepath(token))
            ego_poses.append(self.nusc.get_ego_pose(token))

        # transform ego poses and get state vectors
        _, _, state_vector_list, T_g_s, T_s_g, yaw_g_s = self.nusc.transform_ego_poses(ego_poses,
                                                                self.obs_horizon-1)
        
        if len(image_paths) != self.pred_horizon:
            print('ERROR image', len(image_paths), ' ', self.pred_horizon, ' ', idx)
        if len(state_vector_list) != self.pred_horizon:
            print('ERROR state', len(state_vector_list), ' ', self.pred_horizon, ' ', idx)
        
        # get vehicle data
        scene_token = self.nusc.nusc.get('sample', inital_frame_token)['scene_token']
        scene_name = self.nusc.nusc.get('scene', scene_token)['name']
        vehicle_data = self.nusc_can.get_messages(scene_name,'vehicle_monitor')
        # pad vehicle data such that its length equals samples in scene
        nbr_keyframes = self.nusc.nusc.get('scene', scene_token)['nbr_samples']
        while len(vehicle_data) != nbr_keyframes:
            if len(vehicle_data) < nbr_keyframes:
                vehicle_data.append(vehicle_data[-1])
            else:
                vehicle_data.pop(-1)

        start_index = self.nusc.token_index(nbr_keyframes, inital_frame_token)
        if len(vehicle_data) != nbr_keyframes:
            print('ERROR veh_data', len(vehicle_data), ' ', nbr_keyframes, ' ', start_index)


        cmd_list = []
        for reading in vehicle_data[start_index:start_index+self.pred_horizon]:
            cmd_list.append(self.nusc.get_vel_and_steering_angle(reading))
        
        # pad to right dimensions
        while len(cmd_list) < self.pred_horizon:
            cmd_list.append(cmd_list[-1])
            
        if len(cmd_list) != self.pred_horizon:
            print('ERROR cmd', len(cmd_list), ' ', self.pred_horizon, ' ', idx)

        # get actions
        sample = dict()
        state_vector_list = [position_normalizer(action) for action in state_vector_list]
        cmd_list = [vel_steering_normalizer(cmd) for cmd in cmd_list]
        images = [image_transform(image_path) for image_path in image_paths[:self.obs_horizon]]
        sample['image'] = torch.stack(images)
        sample['agent_pos'] = torch.stack(cmd_list[:self.obs_horizon])
        sample['action'] = torch.stack(cmd_list)
        sample['gt_pos'] = torch.stack(state_vector_list)
        sample['meta_sequence'] = {'sample_token': inital_frame_token, 'T_g_s': T_g_s,
                                   'T_s_g': T_s_g, 'yaw_g_s': yaw_g_s}
        return sample