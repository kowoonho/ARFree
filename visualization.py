from einops import rearrange
import os
import os.path as osp
import mediapy as media
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm
from PIL import Image
import imageio

def visualize(save_path, origin, result, cond_frame_num=4, save_num = 256, save_gif=False, save_each_frame=False):
    """
    Args:
        origin : [B, T, C, H, W]
        result : [B, T, C, H, W]
    """
    
    assert origin.shape == result.shape, f"origin ({origin.shape}) and result ({result.shape}) shape are not equal."
    assert cond_frame_num <= origin.shape[1], f"cond_frame_num ({cond_frame_num}) is too big for video length ({origin.shape[1]})."
    
    
    save_pic_num = min(len(origin), save_num)
    
    print(f"save {save_pic_num} samples")
    
    if save_gif:
        save_gif_path = osp.join(save_path, 'sample_gif')
        os.makedirs(save_gif_path, exist_ok=True)
        
        origin_output = rearrange(origin, 'b t c h w -> b t h w c').detach().cpu().numpy()
        result_output = rearrange(result, 'b t c h w -> b t h w c').detach().cpu().numpy()
        for i in range(save_pic_num):
            combined_gif = []
            for t in range(origin_output.shape[1]):
                combined_frame = np.concatenate((origin_output[i, t], result_output[i, t]), axis=1)
                
                combined_gif.append(combined_frame)
            combined_gif = np.array(combined_gif)
            
            media.write_video(osp.join(save_gif_path, f'sample_{i}.gif'), combined_gif, codec='gif', fps=5)
        
    if save_each_frame:
        save_pic_path = osp.join(save_path, 'sample_pic')
        os.makedirs(save_pic_path, exist_ok=True)
        
        origin_output = rearrange(origin, 'b t c h w -> b t h w c').detach().cpu().numpy()
        result_output = rearrange(result, 'b t c h w -> b t h w c').detach().cpu().numpy()
        for i in range(save_pic_num):
            save_dir = osp.join(save_pic_path, f'sample_{i}')
            os.makedirs(save_dir, exist_ok=True)
            for t in range(origin_output.shape[1]):
                combined_frame = np.concatenate((origin_output[i, t], result_output[i, t]), axis=1)
                # media.write_image(osp.join(save_dir, f'sample_origin_{t}.png'), origin_output[i, t])
                # media.write_image(osp.join(save_dir, f'sample_result_{t}.png'), result_output[i, t])
                media.write_image(osp.join(save_dir, f'sample_combined_{t}.png'), combined_frame)
                
    
    save_path = osp.join(save_path, 'sample_each_image')
    os.makedirs(save_path, exist_ok=True)
    
    origin_output = rearrange(origin, 'b t c h w -> b t h w c').detach().cpu().numpy()
    result_output = rearrange(result, 'b t c h w -> b t h w c').detach().cpu().numpy()
    for i in range(save_pic_num):
        save_dir = osp.join(save_path, f'sample_{i}')
        os.makedirs(save_dir, exist_ok=True)
        for t in range(origin_output.shape[1]):
            media.write_image(osp.join(save_dir, f'origin_{t}.png'), origin_output[i, t])
            media.write_image(osp.join(save_dir, f'result_{t}.png'), result_output[i, t])
    
                
                

def visualize_motion(save_path, origin_vids, pred_vids):
    
    B, T, C, H, W = origin_vids.shape
    
    save_dir = osp.join(save_path, 'sample_warp_with_template')
    os.makedirs(save_dir, exist_ok=True)
    for i in range(B):
        combined_origin_vid = rearrange(origin_vids[i], 't c h w -> c h (t w)')
        combined_pred_vid = rearrange(pred_vids[i], 't c h w -> c h (t w)')
        
        combined_result = torch.cat([combined_origin_vid, combined_pred_vid], dim=1)
        combined_result = rearrange(combined_result, 'c h w -> h w c').detach().cpu().numpy()
    
        media.write_image(osp.join(save_dir, f'sample_{i}.png'), combined_result)
        
    