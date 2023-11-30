import torch
from torch.utils.data import Dataset

from helpers.custom_nuscenes import CustomNuScenes
from helpers.utils import image_transform, position_normalizer


class PositionDataset(Dataset):
    def __init__(self, dataset_path, version, obs_horizon, pred_horizon,
                  action_horizon, test=False, test_tokens_file=None):
        self.nusc = CustomNuScenes(dataset_path, version)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.test = test

        if self.test:
            # use tokens in the test set
            # when sampling, read images with tokens from the test set
            with open(test_tokens_file, 'r') as f:
                self.start_tokens = f.readlines()
                self.start_tokens = [token.strip() for token in self.start_tokens]
        else:
            # use tokens in the training set
            # when sampling, read images with tokens from the training set
            self.scenes = self.nusc.nusc.scene
            # define possible start tokens that can be sampled from
            self.start_tokens = []
            for scene in self.scenes:
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
        # get images and ego_poses
        image_paths = []
        ego_poses = []
        for token in tokens_in_sequence:
            image_paths.append(self.nusc.get_front_cam_filepath(token))
            ego_poses.append(self.nusc.get_ego_pose(token))
        
        # transform ego poses and get state vectors
        _, _, state_vector_list, T_g_s, T_s_g, yaw_g_s = self.nusc.transform_ego_poses(ego_poses,
                                                                self.obs_horizon-1)
        # get actions
        sample = dict()
        state_vector_list = [position_normalizer(action) for action in state_vector_list]
        images = [image_transform(image_path) for image_path in image_paths[:self.obs_horizon]]
        sample['image'] = torch.stack(images)
        sample['agent_pos'] = torch.stack(state_vector_list[:self.obs_horizon])
        sample['action'] = torch.stack(state_vector_list)
        sample['meta_sequence'] = {'sample_token': inital_frame_token, 'T_g_s': T_g_s,
                                    'T_s_g': T_s_g, 'yaw_g_s': yaw_g_s}
        return sample