from torch.utils.data import DataLoader

from torchvision import transforms
from datasets.transform import *


def build_dataset(config):
    transform = transforms.Compose([
        VidResize((config.dataset_params.frame_shape, config.dataset_params.frame_shape)),
        VidToTensor(), 
        ])
    
    if config.type == 'KTH':
        from datasets.dataset import KTHDataset
        train_dataset, val_dataset = KTHDataset(
            KTH_dir = config.dataset_params.data_dir,
            transform = transform,
            num_observed_frames_train = config.train_params.cond_frames,
            num_predict_frames_train = config.train_params.pred_frames,
            num_observed_frames_val = config.valid_params.cond_frames,
            num_predict_frames_val = config.valid_params.pred_frames,
            max_temporal_distance = config.direct.max_temporal_distance,
            val_total_videos = config.valid_params.total_videos,
            consistency_pred = config.train_params.get('consistency_pred', False),
        )()
    elif config.type == 'NATOPS':
        from datasets.dataset import NATOPSDataset
        train_dataset, val_dataset = NATOPSDataset(
            data_dir=config.dataset_params.data_dir,
            transform=transform,
            num_observed_frames_train=config.train_params.cond_frames,
            num_predict_frames_train=config.train_params.pred_frames,
            num_observed_frames_val=config.valid_params.cond_frames,
            num_predict_frames_val=config.valid_params.pred_frames,
            max_temporal_distance=config.direct.max_temporal_distance,
            total_videos=config.valid_params.total_videos,
            consistency_pred=config.train_params.get('consistency_pred', False),
        )()
    else:
        ValueError('Dataset type not supported')
    
    return train_dataset, val_dataset



def build_dataloader(config, train_dataset, val_dataset):
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_params.batch_size,
        shuffle=True,
        num_workers=config.train_params.dataloader_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.valid_params.batch_size,
        shuffle=False,
        num_workers=config.train_params.dataloader_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_dataloader, val_dataloader

