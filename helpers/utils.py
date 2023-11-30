import torch
from PIL import Image
from torchvision import transforms


def image_transform(image_path):
    '''
    Normalizes and transforms the image to a tensor
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] # from imagenet
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(image)

def image_unnormalizer(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] # from imagenet
    for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    return tensor

def position_normalizer(position):
    '''
    Normalizes the positions and transforms the position to a tensor
    '''
    # values calculated over all samples in the dataset
    min_x = -10.078815400284952
    max_x = 126.46777587309384
    min_y = -44.04983327602986
    max_y = 51.734212406292954
    min_yaw = -2.690288779201998
    max_yaw = 2.234569475912055
    position[0] = 2*(position[0] - min_x) / (max_x - min_x) - 1
    position[1] = 2*(position[1] - min_y) / (max_y - min_y) - 1
    position[2] = 2*(position[2] - min_yaw) / (max_yaw - min_yaw) - 1
    return torch.FloatTensor(position)

def position_unnormalizer(tensor):
    min_vals = torch.tensor([-10.078815400284952, -44.04983327602986, -2.690288779201998])
    max_vals = torch.tensor([126.46777587309384, 51.734212406292954, 2.234569475912055])
    
    min_vals = min_vals[:2].unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, 2)
    max_vals = max_vals[:2].unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, 2)
    
    tensor = (tensor[..., :2] + 1) * (max_vals - min_vals) / 2 + min_vals
    return tensor

def vel_steering_normalizer(cmds):
    min_vel = 0
    max_vel = 60/3.6 # convert to m/s
    min_steering = -0.552
    max_steering = 0.552
    cmds[0] = 2*(cmds[0] - min_vel) / (max_vel - min_vel) - 1
    cmds[1] = 2*(cmds[1] - min_steering) / (max_steering - min_steering) - 1
    return torch.FloatTensor(cmds)

def vel_steering_unnormalizer(tensor):
    min_vals = torch.tensor([0, -0.552])
    max_vals = torch.tensor([60/3.6, 0.552])
    min_vals = min_vals.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, 2)
    max_vals = max_vals.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, 2)
    tensor = (tensor + 1) * (max_vals - min_vals) / 2 + min_vals
    return tensor

def vehicle_dynamics(x, u, dt):
    '''
    param: x: state vector (shape: (batch_size, 3))
    param: u: control vector (shape: (batch_size, 2))
    param: dt: time step
    return: x_new: new state vector (shape: (batch_size, 3))
    '''
    wheelbase = 2.588  # Renault Zoe
    x_new = torch.zeros_like(x, dtype=torch.float32)
    x_new[:, 0] = x[:, 0] + u[:, 0] * dt * torch.cos(x[:, 2])
    x_new[:, 1] = x[:, 1] + u[:, 0] * dt * torch.sin(x[:, 2])
    x_new[:, 2] = x[:, 2] + u[:, 0] * dt * torch.tan(u[:, 1]) / wheelbase
    return x_new

def from_norm_cmds_to_positions(cmds):
    '''
    return: unnormalized_positions: tensor of unnormalized positions [x, y, z, 1]
    '''
    batch_size, pred_horizon, _ = cmds.shape
    unnorm_cmds = vel_steering_unnormalizer(cmds)  # Assuming vel_steering_unnormalizer handles tensor shapes
    
    unnormalized_positions = []
    x = torch.zeros(batch_size, 3, dtype=torch.float32)  # Initial state tensor
    delta_t = 0.5

    for u in unnorm_cmds.unbind(1):
        x = vehicle_dynamics(x, u, delta_t)
        unnormalized_positions.append(torch.cat([x, torch.ones((batch_size, 1), dtype=torch.float32)], dim=1))

    return torch.stack(unnormalized_positions, dim=1)

def transform_positions_to_global_frame(positions, T_s_g):
    '''
    param: positions: tensor of positions in the sensor frame (shape: (batch_size, pred_horizon, 2))
    param: T_s_g: transformation matrix from sequence frame to global frame (shape(4,4))
    return: positions_global: tensor of positions in the global frame (shape: (batch_size, pred_horizon, 2))
    '''
    # Convert positions to homogeneous coordinates
    ones_column = torch.ones((*positions.shape[:-1], 1), dtype=positions.dtype, device=positions.device)
    zeros_column = torch.zeros_like(ones_column)
    positions_homo = torch.cat([positions, zeros_column, ones_column], dim=-1)

    # Reshape the transformation matrix if needed
    if T_s_g.shape != (4, 4):
        raise ValueError("Transformation matrix should have shape (4, 4)")

    # # Convert T_s_g to a PyTorch tensor
    # _T_s_g = torch.tensor(T_s_g, dtype=positions.dtype, device=positions.device)
    # stack _T_s_g to match positions batch_size
    # Perform transformation to global frame
    global_positions = []
    for pos in positions_homo:
        global_positions.append(torch.tensor((T_s_g.numpy() @ pos.numpy())[:2]))

    # Convert back to Cartesian coordinates
    #positions_global = positions_global_homo[..., :2] / positions_global_homo[..., 3].unsqueeze(-1)

    return torch.stack(global_positions)