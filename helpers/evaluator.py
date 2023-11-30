import torch
import numpy as np
import json
from helpers.utils import transform_positions_to_global_frame
from nuscenes.prediction import PredictHelper

class Evaluator():
    def __init__(self, nusc):
        self.pred_helper = PredictHelper(nusc)
        self.entries = 0
        # nuscenes specific
        self.start_index = 3
        self.end_index = self.start_index + 6
        self.collisions_1s = 0
        self.collisions_2s = 0
        self.collisions_3s = 0
        self.l2_1s = 0
        self.l2_2s = 0
        self.l2_3s = 0    

    def calc_metrics(self, gt, pred, sample_token, T_s_g):
        '''
        Calculates all metrics
        param: pred: tensor of shape (pred_horizon, 4)
        param: gt: tensor of shape (pred_horizon, 2)
        '''
        
        # adjust lengths accordnig to nuscenes
        pred = pred[self.start_index:self.end_index, :2]
        gt = gt[self.start_index:self.end_index, :]
        # transform to global frame
        pred = transform_positions_to_global_frame(pred, T_s_g)
        gt = transform_positions_to_global_frame(gt, T_s_g)
        # update collisions
        self._update_collisions(pred, sample_token)
        # calculate l2 distance
        self._update_l2_distance(pred, gt)
        self.entries += 1


    def save_metrics(self, path):
        '''
        Saves metrics to file
        '''
        with open(path, 'w') as f:
            f.write('L2 distance for 1 second: ' + str(self.get_l2_1s()) + '\n')
            f.write('L2 distance for 2 seconds: ' + str(self.get_l2_2s()) + '\n')
            f.write('L2 distance for 3 seconds: ' + str(self.get_l2_3s()) + '\n')
            f.write('Collisions for 1 second: ' + str(self.get_collisions_1s()) + '\n')
            f.write('Collisions for 2 seconds: ' + str(self.get_collisions_2s()) + '\n')
            f.write('Collisions for 3 seconds: ' + str(self.get_collisions_3s()) + '\n')


    def _update_l2_distance(self, pred, gt):
        '''
        Calculates the l2 distance between two tensors
        '''
        ego_trajectory = pred
        ego_traj_1s, ego_traj_2s, ego_traj_3s = self._get_1_2_3_sequences(ego_trajectory)
        # gt
        gt_1s, gt_2s, gt_3s = self._get_1_2_3_sequences(gt)
        # calculate l2 distance
        l2_1s = torch.norm(ego_traj_1s - gt_1s, p=2, dim=1)
        l2_2s = torch.norm(ego_traj_2s - gt_2s, p=2, dim=1)
        l2_3s = torch.norm(ego_traj_3s - gt_3s, p=2,dim=1)
        # update metrics
        self.l2_1s += torch.sum(l2_1s, dim=-1)/ego_traj_1s.shape[0]
        self.l2_2s += torch.sum(l2_2s, dim=-1)/ego_traj_2s.shape[0]
        self.l2_3s += torch.sum(l2_3s, dim=-1)/ego_traj_3s.shape[0]



    def _update_collisions(self, pred, sample_token):
        '''
        Updates the collision metrics
        '''
        # get all annotation tokens in sample
        ann_tokens = self._get_ann_tokens_in_sample(sample_token)
        # get all trajectories for annotations
        trajectories = self._get_trajectories_for_ann_tokens(ann_tokens)
        # get ego trajectory
        ego_trajectory = pred
        ego_traj_1s, ego_traj_2s, ego_traj_3s = self._get_1_2_3_sequences(ego_trajectory)
        # check if collided
        if self._check_if_collided(ego_traj_1s, trajectories):
            self.collisions_1s += 1
            self.collisions_2s += 1
            self.collisions_3s += 1
            return
        if self._check_if_collided(ego_traj_2s, trajectories):
            self.collisions_2s += 1
            self.collisions_3s += 1
            return
        if self._check_if_collided(ego_traj_3s, trajectories):
            self.collisions_3s += 1
            return

    def _get_1_2_3_sequences(self, sequence):
        '''
        Returns the 1, 2 and 3 second sequences
        '''
        return sequence[:2], sequence[:4], sequence[:6]


    def _check_if_collided(self, ego_trajectory, obj_trajectories):
        '''
        Returns the number of collisions between ego and obj trajectories
        '''
        delta_x = 3.0
        delta_y = 1.5
        collided = False
        for obj_trajectory in obj_trajectories:
            for obj_pos, ego_pos in zip(obj_trajectory, ego_trajectory):
                if abs(obj_pos[0] - ego_pos[0]) < delta_x and abs(obj_pos[1] - ego_pos[1]) < delta_y:
                    collided = True
                    return collided
        return collided
    
    def _get_ann_tokens_in_sample(self, sample_token):
        '''
        Returns all annotations in sample
        '''
        sample = self.pred_helper.data.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']
        return sample_annotation_tokens
    
    def _get_trajectories_for_ann_tokens(self, ann_tokens, time_horizon=3):
        '''
        Returns all trajectories for a list of annotation tokens
        '''
        trajectories = []
        for ann_token in ann_tokens:
            object = self.pred_helper.data.get('sample_annotation', ann_token)
            positions = self.pred_helper.get_future_for_agent(object['instance_token'],
                                                              object['sample_token'],
                                                                time_horizon,
                                                                in_agent_frame=False,
                                                                just_xy=True)
            trajectories.append(positions)
        
        return trajectories

    def get_l2_1s(self):
        '''
        Returns the l2 distance for 1 second
        '''
        return (self.l2_1s/self.entries).numpy()
    def get_l2_2s(self):
        '''
        Returns the l2 distance for 2 seconds
        '''
        return (self.l2_2s/self.entries).numpy()
    def get_l2_3s(self):
        '''
        Returns the l2 distance for 3 seconds
        '''
        return (self.l2_3s/self.entries).numpy()
    def get_collisions_1s(self):
        '''
        Returns the number of collisions for 1 second
        '''
        return self.collisions_1s/self.entries
    def get_collisions_2s(self):
        '''
        Returns the number of collisions for 2 seconds
        '''
        return self.collisions_2s/self.entries
    def get_collisions_3s(self):
        '''
        Returns the number of collisions for 3 seconds
        '''
        return self.collisions_3s/self.entries
    