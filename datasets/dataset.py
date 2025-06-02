from torch.utils import data
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms as T, utils
import torchvision.transforms.functional as F
from functools import partial
import random
import torchvision.transforms as transforms
import cv2
import numpy as np
from einops import rearrange

import os


CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

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

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class KTHDataset(object):
    """
    KTH dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (120, 160)
    Split the KTH dataset and return the train and test dataset
    """
    def __init__(self, KTH_dir, transform,
                 num_observed_frames_train, num_predict_frames_train, num_observed_frames_val, num_predict_frames_val,
                 max_temporal_distance=24, val_total_videos=0, is_chunk=True,
                 consistency_pred=False,
                 ):
        np.random.seed(0)
        """
        Args:
            KTH_dir --- Directory for extracted KTH video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
        """
        self.num_observed_frames_train = num_observed_frames_train
        self.num_predict_frames_train = num_predict_frames_train
        self.num_observed_frames_val = num_observed_frames_val
        self.num_predict_frames_val = num_predict_frames_val
        self.max_temporal_distance = max_temporal_distance
        
        self.val_total_videos = val_total_videos
        self.is_chunk = is_chunk
        self.consistency_pred = consistency_pred
        
        self.total_length_train = self.max_temporal_distance + self.num_observed_frames_train
        
        
        if self.consistency_pred:
            self.num_predict_frames_train += self.num_predict_frames_train - 1
            # self.max_temporal_distance -= self.num_predict_frames_train
        
        
        self.clip_length_train = num_observed_frames_train + num_predict_frames_train
        self.clip_length_val = num_observed_frames_val + num_predict_frames_val
        
        self.transform = transform
        self.color_mode = 'RGB'
        self.actions = {'boxing':0, 'handclapping':1, 'handwaving':2, 'jogging_no_empty':3, 'running_no_empty':4, 'walking_no_empty':5}

        self.KTH_path = Path(KTH_dir).absolute()
        self.person_ids, self.val_person_ids = list(range(1, 21)), list(range(21, 26))

        frame_folders = self.__getFramesFolder__(self.person_ids)
        self.video_data = self.__getTrainData__(frame_folders)
        
        val_frame_folders = self.__getFramesFolder__(self.val_person_ids)
        self.val_video_data = self.__getValData__(val_frame_folders)
            
        

    def __call__(self):
        train_dataset = ClipTrainDataset(self.num_observed_frames_train, self.num_predict_frames_train, self.video_data, self.transform, self.color_mode)
        val_dataset = ClipValDataset(self.num_observed_frames_val, self.num_predict_frames_val, self.val_video_data, self.val_total_videos, self.transform, self.color_mode)
        return train_dataset, val_dataset
    
    def generate_random_chunk(self, start_timestep):
        
        selected_timestep = [i for i in range(start_timestep, start_timestep + self.num_observed_frames_train)]
        
        pred_start = random.sample(range(start_timestep + self.num_observed_frames_train, start_timestep + self.num_observed_frames_train + self.max_temporal_distance - self.num_predict_frames_train + 1), 1)[0]
        
        selected_timestep.extend([i for i in range(pred_start, pred_start + self.num_predict_frames_train)])
        
        return selected_timestep
    
    def __getTrainData__(self, frame_folders):
        clips = []
        frame_indices = []
        action_classes = []
        
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            action_class = self.actions[folder.parent.name]
            
            for i in range(0, len(img_files) - (self.total_length_train) + 1):
                if i >= len(img_files) - self.clip_length_train: break
                
                if self.max_temporal_distance == 0:
                    selected_timestep = [i+j for j in range(self.clip_length_train)]
                elif self.is_chunk:
                    selected_timestep = self.generate_random_chunk(i)
                else:
                    selected_timestep = sorted(random.sample(range(i, i + self.max_temporal_distance), self.clip_length_train))

                frame_idx = selected_timestep
                clip = [img_files[t] for t in selected_timestep]
                frame_idx = [t - min(selected_timestep) for t in selected_timestep]
                
                clips.append(clip)
                frame_indices.append(frame_idx)
                action_classes.append(action_class)

        return {'clips':clips, 'frame_indices':frame_indices, 'action_classes':action_classes}
    
    def __getValData__(self, frame_folders):
        clips = []
        action_classes = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            if len(img_files) < self.num_predict_frames_val + self.num_observed_frames_val:
                continue
            action_class = self.actions[folder.parent.name]
            
            cond_timestep = np.random.randint(0, max(len(img_files) - self.num_predict_frames_val - self.num_observed_frames_val, 0) + 1)
            clip = img_files[cond_timestep:min(cond_timestep + self.num_predict_frames_val + self.num_observed_frames_val, len(img_files))]
            clips.append(clip)
            action_classes.append(action_class)
            
        return {'clips':clips, 'action_classes':action_classes}
    
    def __getFramesFolder__(self, person_ids):
        """
        Get the KTH frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []
        for a in self.actions.keys():
            action_path = self.KTH_path.joinpath(a)
            frame_folders.extend([action_path.joinpath(s) for s in os.listdir(action_path) if '.avi' not in s and s.startswith('person')])
        frame_folders = sorted(frame_folders)
        
        return_folders = []
        for ff in frame_folders:
            person_id = int(str(ff.name).strip().split('_')[0][-2:])
            if person_id in person_ids:
                return_folders.append(ff)

        return return_folders


class ClipTrainDataset(data.Dataset):
    """
    Video clips dataset
    """
    def __init__(self, num_observed_frames, num_predict_frames, video_data, transform, color_mode='RGB',
                 use_crop=False, mean=False, resize=False):
        """
        Args:
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clips --- List of video clips frames file path
            transfrom --- torchvision transforms for the image
            color_mode --- 'RGB' for RGB dataset, 'grey_scale' for grey_scale dataset

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_observed_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_predict_frames, C, H, W)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.video_data = video_data
        self.transform = transform
        if color_mode != 'RGB' and color_mode != 'grey_scale':
            raise ValueError("Unsupported color mode!!")
        else:
            self.color_mode = color_mode
        self.use_crop = use_crop
        self.mean = mean
        self.resize = resize

    def __len__(self):
        return len(self.video_data['clips'])
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_observed_frames, C, H, W)
            future_clip: Tensor with shape (num_predict_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_img_paths = self.video_data['clips'][index]
        frame_indices = self.video_data['frame_indices'][index]
        action = self.video_data['action_classes'][index]
        
        imgs = []
        for img_path in clip_img_paths:
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(np.array(img))
        
        if self.use_crop:
            imgs = [img[10:239, 30:290, :] for img in imgs]
        if self.resize:
            imgs = [resize(img, 128, interpolation=cv2.INTER_AREA) for img in imgs]    
        
        if self.mean:
            imgs = [img - (0., 0., 0.) for img in imgs]
        imgs = [Image.fromarray(img.astype(np.uint8)) for img in imgs]
            
        original_clip = rearrange(self.transform(imgs), 't c h w -> c t h w')

        # past_clip = original_clip[:, 0:self.num_observed_frames]
        # future_clip = original_clip[:, -self.num_predict_frames:]
        
        return original_clip, torch.tensor(frame_indices, dtype=torch.long), torch.tensor(action, dtype=torch.long)

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)
        
        videodims = img.size
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')    
        video = cv2.VideoWriter(Path(file_name).absolute().as_posix(), fourcc, 10, videodims)
        for img in imgs:
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        video.release()
        #imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])
        
class ClipValDataset(data.Dataset):
    def __init__(self, num_observed_frames, num_predict_frames, video_data, total_videos=0, transform=None, color_mode='RGB',
                 use_crop=False, mean=False, resize=False):
        np.random.seed(0)
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.video_data = video_data
        self.transform = transform
        self.color_mode = color_mode
        self.total_videos = total_videos
        
        self.use_crop = use_crop
        self.mean = mean
        self.resize = resize
        
        if total_videos != 0:
            self.selected_idx = sorted(np.random.choice(len(self.video_data['clips']), total_videos, replace=False))
            
        
    def __len__(self):
        return self.total_videos if self.total_videos != 0 else len(self.video_data['clips'])
    
    def __getitem__(self, index):
        index = self.selected_idx[index]
        clip_img_paths = self.video_data['clips'][index]
        action = self.video_data['action_classes'][index]
        
        imgs = []
        for img_path in clip_img_paths:
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(np.array(img))
            
        if self.use_crop:
            imgs = [img[10:239, 30:290, :] for img in imgs]
        if self.resize:
            imgs = [resize(img, 128, interpolation=cv2.INTER_AREA) for img in imgs]    
        
        if self.mean:
            imgs = [img - (0., 0., 0.) for img in imgs]
        imgs = [Image.fromarray(img.astype(np.uint8)) for img in imgs]
        
        video = rearrange(self.transform(imgs), 't c h w -> c t h w')
        return video, torch.tensor(action, dtype=torch.long)
    

class NATOPSDataset(object):
    def __init__(
        self, 
        data_dir, 
        transform, 
        num_observed_frames_train, 
        num_predict_frames_train,
        num_observed_frames_val,
        num_predict_frames_val,
        max_temporal_distance=30,
        is_chunk=True,
        consistency_pred=False,
        total_videos=0,
    ):
        np.random.seed(0)
        self.data_dir = Path(data_dir).absolute()
        self.num_observed_frames_train = num_observed_frames_train
        self.num_predict_frames_train = num_predict_frames_train
        self.num_observed_frames_val = num_observed_frames_val
        self.num_predict_frames_val = num_predict_frames_val
        self.max_temporal_distance = max_temporal_distance
        self.is_chunk = is_chunk
        self.consistency_pred = consistency_pred
        self.total_videos = total_videos
        
        self.total_length_train = self.max_temporal_distance + self.num_observed_frames_train
        
        if self.consistency_pred:
            self.num_predict_frames_train += self.num_predict_frames_train - 1
        
        self.clip_length_train = num_observed_frames_train + num_predict_frames_train
        self.clip_length_val = num_observed_frames_val + num_predict_frames_val
        
        self.transform = transform
        
        # By LFDM
        self.train_subject_ids = [3, 4, 8, 9, 12, 13, 15, 17, 19, 20]
        self.val_subject_ids = [1, 2, 5, 6, 7, 10, 11, 14, 16, 18]
        
        # Load training and validation data
        train_frame_folders = self.__getFramesFolder__(self.train_subject_ids)
        self.train_data = self.__getTrainData__(train_frame_folders)
        val_frame_folders = self.__getFramesFolder__(self.val_subject_ids)
        self.val_data = self.__getValData__(val_frame_folders)
    
    def __call__(self):
        train_dataset = ClipTrainDataset(self.num_observed_frames_train, self.num_predict_frames_train, self.train_data, self.transform,
                                         use_crop=True, mean=True, resize=True)
        val_dataset = ClipValDataset(self.num_observed_frames_val, self.num_predict_frames_val, self.val_data, 
                                     transform=self.transform, total_videos=self.total_videos,
                                     use_crop=True, mean=True, resize=True)
        return train_dataset, val_dataset
        
    def generate_random_chunk(self, start_timestep):
        
        selected_timestep = [i for i in range(start_timestep, start_timestep + self.num_observed_frames_train)]
        
        pred_start = random.sample(range(start_timestep + self.num_observed_frames_train, start_timestep + self.num_observed_frames_train + self.max_temporal_distance - self.num_predict_frames_train + 1), 1)[0]
        
        selected_timestep.extend([i for i in range(pred_start, pred_start + self.num_predict_frames_train)])
        
        return selected_timestep
    
    def __getTrainData__(self, frame_folders):
        clips = []
        frame_indices = []
        action_classes = []
        
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            action_class = int(folder.name[1:3]) - 1
            
            
            if len(img_files) >= self.total_length_train:
                sample_idx_list = np.linspace(start=0, stop=len(img_files)-1, num=self.total_length_train, dtype=int)
            else:
                sample_idx_list = np.linspace(start=0, stop=len(img_files)-1, num=len(img_files), dtype=int)
            
            if len(sample_idx_list) < self.clip_length_train:
                continue
            for i in range(self.num_observed_frames_train, len(sample_idx_list) - self.num_predict_frames_train + 1):
                
                cond_timestep = [k for k in range(self.num_observed_frames_train)]
                
                pred_timestep = [k for k in range(i, i + self.num_predict_frames_train)]
                
                frame_idx = cond_timestep + pred_timestep
                clip = [img_files[sample_idx_list[t]] for t in frame_idx]
                
                clips.append(clip)
                frame_indices.append(frame_idx)
                action_classes.append(action_class)
                
        return {'clips':clips, 'frame_indices':frame_indices, 'action_classes':action_classes}

    def __getValData__(self, frame_folders):
        clips = []
        action_classes = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            if len(img_files) < self.num_predict_frames_val + self.num_observed_frames_val:
                continue
            action_class = int(folder.name[1:3]) - 1
            
            sample_idx_list = np.linspace(start=0, stop=len(img_files)-1, num=self.total_length_train, dtype=int)
            clip = [img_files[sample_idx_list[t]] for t in range(0, len(sample_idx_list))]            
            clips.append(clip)
            action_classes.append(action_class)
            
        return {'clips':clips, 'action_classes':action_classes}
    
    def __getFramesFolder__(self, subject_ids):
        """
        Get the NATOPS frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []        
        for folder in os.listdir(self.data_dir):
            action = folder[1:3]
            subject = int(folder[4:6])
            person = int(folder[7:9])
            
            if subject in subject_ids:
                frame_folders.append(self.data_dir.joinpath(folder))
            
        return frame_folders
                
        