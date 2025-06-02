from torchvision import transforms
from typing import List
from PIL import Image
from torch import Tensor
import torch
import numpy as np

class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.resize_kwargs['antialias'] = True
        self.resize_kwargs['interpolation'] = transforms.InterpolationMode.BICUBIC
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Resize(*self.args, **self.resize_kwargs)(clip[i])

        return clip

class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.CenterCrop(*self.args, **self.kwargs)(clip[i])

        return clip

class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.functional.crop(clip[i], *self.args, **self.kwargs)

        return clip
        
class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip

class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.vflip(clip[i])
        return clip

class VidToTensor(object):
    def __call__(self, clip: List[Image.Image]):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim = 0)

        return clip

class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = transforms.Normalize(self.mean, self.std)(clip[i, ...])

        return clip

class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0/s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.],
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = [1., 1., 1.])])
        except TypeError:
            #try normalize for grey_scale images.
            self.inv_std = 1.0/std
            self.inv_mean = -mean
            self.renorm = transforms.Compose([transforms.Normalize(mean = 0.,
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = 1.)])

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

        return clip