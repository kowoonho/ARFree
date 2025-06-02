# loading video dataset for training and testing
import os
import torch

import numpy as np
import torch.utils.data as data

import cv2


from einops import rearrange, repeat

def dataset2video(video):
    if len(video.shape) == 3:
        video = repeat(video, 't h w -> t c h w', c=3)
    elif video.shape[1] == 1:
        video = repeat(video, 't c h w -> t (n c) h w', n=3)
    else:
        video = rearrange(video, 't h w c -> t c h w')
    return video

def dataset2videos(videos):
    if len(videos.shape) == 4:
        videos = repeat(videos, 'b t h w -> b t c h w', c=3)
    elif videos.shape[2] == 1:
        videos = repeat(videos, 'b t c h w -> b t (n c) h w', n=3)
    else:
        videos = rearrange(videos, 'b t h w c -> b t c h w')
    return videos
        
def resize(im, desired_size, interpolation):
    old_size = im.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple(int(x*ratio) for x in old_size)

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=interpolation)
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im

class DatasetRepeater(data.Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]

if __name__ == "__main__":
    pass
    # check_video_data_structure()
    # check_num_workers()
    

