import argparse
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from helpers.utils import *
from datasets.position_dataset import PositionDataset
from datasets.vel_steering_dataset import VelSteeringDataset
from models.modified_resnet import *
from models.modified_unet import *

def train(action_type, model):
    # define horizons
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
        # define dataset
        dataset = PositionDataset(dataset_path, version, obs_horizon, pred_horizon, action_horizon)
    elif action_type == 'vel_steer':
        action_dim = 2
        lowdim_obs_dim = action_dim
        dataset = VelSteeringDataset(dataset_path, version, obs_horizon, pred_horizon, action_horizon)
    
    else:
        print('ERROR: non existing action type:', action_type)
        exit()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
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

    if model != 'empty':
        print('Loading pretrained model')
        checkpoint = torch.load(model, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        nets.load_state_dict(model_state_dict)
        epoch_start = checkpoint['epoch']
        print('Model loaded from', model)
    else:
        epoch_start = 0
        print('No pretrained model loaded, starting from scratch')
    
    print('Model constructed and moved to device')
    # training loop
    num_epochs = 100 - epoch_start
    save_checkpoint_freq = 10
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        model=nets,
        power=0.75,
        parameters=nets.parameters())

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500 if epoch_start == 0 else 0,
        num_training_steps=len(dataloader) * num_epochs
    )
    print('Starting training loop')
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:,:obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            if (epoch_idx + 1) % save_checkpoint_freq == 0:
                with open('loss.txt', 'a') as file:
                    file.write(str(loss_cpu) + '\n')
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': nets.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                }, f'/cluster/work/simenmsa/models/checkpoint_epoch_{epoch_idx + epoch_start + 1}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training flags')
    parser.add_argument('-action', '--action', default='empty', help='positions or vel_steer')
    parser.add_argument('-model', '--model', default='empty', help='model name')
    args = parser.parse_args()
    train(args.action, args.model)