import os
import torch
import numpy as np
import random

from datasets.dataset import unnormalize_img, normalize_img
from torchvision.utils import save_image

def count_parameters(model):
    res = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"count_training_parameters: {res}")
    res = sum(p.numel() for p in model.parameters())
    print(f"count_all_parameters:      {res}")
    
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def save_test_video(video, dir='test_sample', prefix='sample'):
    os.makedirs(dir, exist_ok=True)

    B, C, T, H, W = video.shape
    
    for b in range(B):
        sample_dir = os.path.join(dir, f'{prefix}_{b}')
        os.makedirs(sample_dir, exist_ok=True)
        
        for t in range(T):
            frame = video[b, :, t]
            
            if frame.min() < 0 or frame.max() > 1:
                frame = unnormalize_img(frame)
            save_image(frame, os.path.join(sample_dir, f'frame_{t}.png'))