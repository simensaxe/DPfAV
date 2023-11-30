from torch.utils.data import DataLoader
import numpy as np
import copy
import argparse
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from datasets.position_dataset import PositionDataset
from datasets.vel_steering_dataset import VelSteeringDataset
from models.modified_resnet import *
from models.modified_unet import *
from helpers.evaluator import Evaluator
from helpers.utils import position_unnormalizer, from_norm_cmds_to_positions


def test(action_type, model_path, metrics_save_file, batch_size=1):
    batch_size = int(batch_size)
    obs_horizon = 2
    pred_horizon = 16
    action_horizon = 8
    print('Loading dataset')
    dataset_path = '/cluster/work/simenmsa/nuscenes'
    version = 'v1.0-trainval'
   
    if action_type == 'positions':
        # define dimensions
        action_dim = 3
        # agent_pos is 3 dimensional
        lowdim_obs_dim = action_dim
        dataset = PositionDataset(dataset_path, version, obs_horizon, pred_horizon, action_horizon, test=True, test_tokens_file='test_sample_tokens.txt')
    elif action_type == 'vel_steer':
        action_dim = 2
        lowdim_obs_dim = action_dim
        dataset = VelSteeringDataset(dataset_path, version, obs_horizon, pred_horizon, action_horizon, test=True, test_tokens_file='test_sample_tokens.txt')
    else:
        print('ERROR: non existing action type:', action_type)
        exit()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Dataset loaded, and contains', len(dataset), 'samples')

    print('Constructing model')
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')
    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # ResNet18 has output dim of 512
    vision_feature_dim = 512

    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
    )
    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    # load pretrained weights  
    checkpoint = torch.load(model_path, map_location='cuda')
    model_state_dict = checkpoint['model_state_dict']  # Access the model's state_dict
    ema_nets = nets
    # Load the state_dict into your model
    ema_nets.load_state_dict(model_state_dict)
    print('Pretrained weights loaded.')

    evaluator = Evaluator(dataset.nusc.nusc)
    print('Starting evaluation')
    for batch_idx, data in enumerate(dataloader):
        # already normalized
        images = np.stack(data['image'])
        agent_poses = np.stack(data['agent_pos'])
        images = torch.from_numpy(images).to(device, dtype=torch.float32)
        agent_poses = torch.from_numpy(agent_poses).to(device, dtype=torch.float32)
        with torch.no_grad():
            visual_emb = ema_nets['vision_encoder'](images.flatten(end_dim=1))
            visual_features = visual_emb.reshape(*images.shape[:2], -1)
            obs_cond = torch.cat([visual_features, agent_poses], dim=-1).view(images.shape[0], -1)
            noisy_action = torch.randn((images.shape[0], pred_horizon, action_dim), device=device)
            naction = noisy_action

            noise_scheduler.set_timesteps(num_diffusion_iters)
            for k in noise_scheduler.timesteps:
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond)
                
                # invserse diffusion process
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
                
        naction = naction.detach().cpu()

       
        if action_type == 'positions':
            # transform to positions
            unormalized_positions_pred = position_unnormalizer(naction)
            unorm_pos_gt = position_unnormalizer(data['action'])
            
        else:
            # transform to positions
            unormalized_positions_pred = from_norm_cmds_to_positions(naction)
            unorm_pos_gt = position_unnormalizer(data['gt_pos'])
        # calculate loss
        for pred, batch in zip(unormalized_positions_pred, unorm_pos_gt):
            pred_cp = copy.deepcopy(pred)
            gt_cp = copy.deepcopy(batch)
            evaluator.calc_metrics(gt_cp, pred_cp, data['meta_sequence']['sample_token'][0],
                                data['meta_sequence']['T_s_g'][0])
        
        
        print('Batch', batch_idx, 'done')

    
    print('Evaluation done')
    evaluator.save_metrics(metrics_save_file)
    print('Saved metrics to', metrics_save_file)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test flags')
    parser.add_argument('-action', '--action', default='empty', help='positions or vel_steer')
    parser.add_argument('-model_path', '--model_path', default='empty', help='model name')
    parser.add_argument('-metrics_save_file', '--metrics_save_file', default='empty', help='metrics save file')
    parser.add_argument('-batch_size', '--batch_size', default=1, help='batch size')
    args = parser.parse_args()
    test(args.action, args.model_path, args.metrics_save_file, args.batch_size)
    