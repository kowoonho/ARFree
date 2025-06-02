import argparse
import timeit
import os
import os.path as osp
import math
from tqdm import tqdm

import copy

import numpy as np
import torch
from omegaconf import OmegaConf
from utils.config import load_config
from utils.meter import AverageMeter, RunningAverageMeter
from utils.optimizer import build_optimizer
from utils.checkpoint import load_checkpoint_with_accelerator, save_checkpoint_with_accelerator, save_ema_checkpoint, load_ema_checkpoint

from einops import rearrange, repeat

from datasets.builder import build_dataloader, build_dataset

from utils.utils import count_parameters

from ema_pytorch import EMA

from datasets.video_dataset import DatasetRepeater
from datasets.dataset import normalize_img

import torch.backends.cudnn as cudnn

from accelerate import Accelerator
from accelerate.utils import set_seed

from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate import InitProcessGroupKwargs
from datetime import timedelta

from model.util import noise_sampling, import_module

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/kth.yaml')
    parser.add_argument('--valid', action='store_true')

    
    args = parser.parse_args()
    return args

def main(cfg, valid=False):
    os.makedirs(cfg.train_params.save_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint.output, exist_ok=True)
    
    trainer = Trainer(cfg)
    trainer.train(cfg, valid=valid)
    pass
    
class Trainer(object):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        if cfg.train_params.seed is not None:
            set_seed(cfg.train_params.seed)
            
        ipg_handler = InitProcessGroupKwargs(
                timeout=timedelta(seconds=7200)
                )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.grad_accumulation_steps,
            mixed_precision=cfg.train_params.mixed_precision,
            log_with= "wandb" if cfg.wandb.enable else None,
            kwargs_handlers=[DDPK(find_unused_parameters=True), ipg_handler]
        )
        
        self.model = import_module("model.video_prediction", cfg.model.type)(cfg).to(self.accelerator.device)
        
        if cfg.train_params.ema.enable:
            self.ema_decay = cfg.train_params.ema.decay
            self.ema_start = cfg.train_params.ema.start
            self.ema_update = cfg.train_params.ema.update_freq
            
            self.ema = EMA(self.model, beta=self.ema_decay, update_every=self.ema_update)
            self.ema.to(self.accelerator.device)
            
    def train(self, cfg, valid=False):
        
        # Meter setting
        lr_meter = RunningAverageMeter()
        losses = RunningAverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        
        count_parameters(self.model)
        
        # Data loading
        train_dataset, val_dataset = build_dataset(cfg.dataset)
        
        if cfg.train_params.num_repeats is not None or cfg.train_params.num_repeats > 1:
            train_dataset = DatasetRepeater(train_dataset, cfg.train_params.num_repeats)
            
        train_loader, val_loader = build_dataloader(cfg.dataset, train_dataset, val_dataset)
        
        total_batch_size = cfg.dataset.train_params.batch_size * self.accelerator.num_processes * cfg.train_params.grad_accumulation_steps
        steps_per_epoch = math.ceil(len(train_dataset) / total_batch_size)
        final_step = steps_per_epoch * cfg.train_params.max_epochs
        
        # optimizer & lr_scheduler setting
        optimizer = build_optimizer(cfg.train_params, self.model)
        
        self.model, optimizer, train_loader = self.accelerator.prepare(
            self.model, optimizer, train_loader
        )
        # load checkpoint
        if cfg.checkpoint.resume:
            global_step, epoch_cnt, lr_meter, losses = load_checkpoint_with_accelerator(cfg, self.accelerator, lr_meter, losses)
            if cfg.train_params.ema.enable:
                self.ema = load_ema_checkpoint(cfg, self.ema)
        else:
            global_step = 0
            epoch_cnt = 0
        
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(cfg.project_name)
            if cfg.wandb.enable:
                self.accelerator.trackers[0].run.name = f'{cfg.model_name}_{cfg.dataset.valid_params.cond_frames}_{cfg.dataset.valid_params.pred_frames}'
        
        if valid:
            self.model.eval()
            meters = self.valid(cfg, val_loader, global_step, vis=cfg.visualize)
            exit()
        
        
        
        if self.accelerator.is_main_process:
            print("***** Running training *****")
            print(f"  Num examples = {len(train_dataset)}")
            print(f"  Num Epochs = {cfg.train_params.max_epochs}")
            print(f"  Instantaneous batch size per device = {cfg.dataset.train_params.batch_size}")
            print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            print(f"  Gradient Accumulation steps = {cfg.train_params.grad_accumulation_steps}")
            print(f"  Total optimization steps = {final_step}")
            print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))
        
        progress_bar = tqdm(range(global_step, final_step), disable = not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        
        for epoch in range(epoch_cnt, cfg.train_params.max_epochs):
            for iter, batch in enumerate(train_loader):
                iter_end = timeit.default_timer()
                with self.accelerator.accumulate(self.model):
                    data_time.update(timeit.default_timer() - iter_end)
                    
                    video_frames, frame_indices, action_classes = batch
                    
                    with self.accelerator.autocast():
                        loss_dict = self.model(normalize_img(video_frames), frame_indices, action_classes)
                        total_loss = loss_dict['total_loss']
                        
                    self.accelerator.backward(total_loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 0.3)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    losses.synchronize_and_update(self.accelerator, total_loss, global_step)
                    batch_time.update(timeit.default_timer() - iter_end)
                    iter_end = timeit.default_timer()
                    
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    if self.accelerator.is_main_process and cfg.train_params.ema.enable:
                        self.ema.update()
                    
                    if global_step % cfg.train_params.save_ckpt_freq == 0:
                        save_checkpoint_with_accelerator(cfg, self.accelerator, global_step, epoch, lr_meter, losses)
                        if cfg.train_params.ema.enable:
                            save_ema_checkpoint(cfg, self.ema, global_step, epoch)
                    
                    if global_step % cfg.train_params.valid_freq == 0:
                        if self.accelerator.is_main_process:
                            meters = self.valid(cfg, val_loader, global_step, vis=cfg.visualize)
                            logs = {'FVD': meters['metrics/fvd'], 'SSIM' : meters['metrics/ssim'], 'PSNR' : meters['metrics/psnr'],  'LPIPS' : meters['metrics/lpips'],}
                            self.accelerator.log(logs, step=global_step)
                        self.model.train()
                    
                # log scale loss                        
                logs = {
                    **{key: np.log10(value.detach().item() + 1e-8) for key, value in loss_dict.items()},
                    'lr': optimizer.param_groups[0]['lr']
                }
                progress_bar.set_postfix(logs)
                self.accelerator.log(logs, step=global_step)
                if global_step >= final_step:
                    break
            self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        
    @torch.inference_mode()
    def valid(self, cfg, valid_loader, global_step, vis=None):
        cudnn.enabled=True
        cudnn.benchmark=True
        if cfg.train_params.seed is not None:
            set_seed(cfg.train_params.seed)
        
        if cfg.train_params.ema.enable:
            model = self.ema.ema_model.eval()
        else:
            model = self.accelerator.unwrap_model(self.model).eval()
        model.to(self.accelerator.device)
        
        origin_vids = []
        result_vids = []
        
        total_pred_frames = cfg.dataset.valid_params.pred_frames
        
        from math import ceil
        NUM_ITER = ceil(cfg.dataset.valid_params.total_videos / cfg.dataset.valid_params.batch_size)
        NUM_AUTOREG = ceil(cfg.dataset.valid_params.pred_frames / cfg.dataset.train_params.pred_frames)
        
        cf, pf = cfg.dataset.train_params.cond_frames, cfg.dataset.train_params.pred_frames
        
        for i_iter, batch in enumerate(valid_loader):
            if i_iter >= NUM_ITER: break
            
            video, action = batch
            video, action = video.to(self.accelerator.device), action.to(self.accelerator.device)

            bs = video.shape[0]
            cond_frames = video[:,:, :cf]
            cond_indices = [i for i in range(cf)]

            result_frames = []
            x_noise = noise_sampling(video.shape, self.accelerator.device, cfg.model.diffusion.noise_params)
            
            for step in range(NUM_AUTOREG):
                origin_frames = video[:,:,cf+step*pf:cf+(step+1)*pf]
                frame_indices = torch.tensor(cond_indices + [i for i in range(cf+step*pf, cf+(step+1)*pf)], dtype=torch.long).to(self.accelerator.device)
                frame_indices = repeat(frame_indices, 't -> b t', b=bs)
                
                x = torch.cat([cond_frames, origin_frames], dim=2)

                pred_frames = \
                    model.sample_video(
                        x_noise[:,:,cf+step*pf:cf+(step+1)*pf], normalize_img(x), frame_indices, action
                    )
                
                result_frames.append(pred_frames)
                    

            result_frames = torch.cat(result_frames, dim=2)
            
            origin_vids.append(video[:,:,:cf+total_pred_frames])
            result_vids.append(torch.cat([video[:,:,:cf], result_frames[:,:,:total_pred_frames]], dim=2))
            print(f'[{i_iter+1}/{NUM_ITER}] generated. result_video: {result_vids[-1].shape}')
        
        origin_vids = torch.cat(origin_vids, dim=0) # [B, C, T, H, W]
        result_vids = torch.cat(result_vids, dim=0) # [B, C, T, H, W]
        
        origin_vids = rearrange(origin_vids, 'b c t h w -> b t c h w')
        result_vids = rearrange(result_vids, 'b c t h w -> b t c h w')
        
        if vis is not None:
            from visualization import visualize
            visualize(save_path = osp.join(cfg.train_params.save_dir, 'vis', cfg.model_name), 
                    origin = origin_vids, 
                    result = result_vids, 
                    cond_frame_num = cf, 
                    save_each_frame = vis.save_pic,
                    save_gif = vis.save_gif)

            
        # performance metrics
        from metrics.calculate_fvd import calculate_fvd1
        from metrics.calculate_psnr import calculate_psnr1
        from metrics.calculate_ssim import calculate_ssim1
        from metrics.calculate_lpips import calculate_lpips1
        
        fvd = calculate_fvd1(origin_vids, result_vids, torch.device("cuda"), mini_bs=16)
        videos1 = origin_vids[:, cf:]
        videos2 = result_vids[:, cf:]
        ssim = calculate_ssim1(videos1, videos2)[0]
        psnr = calculate_psnr1(videos1, videos2)[0]
        lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
        print("Total frame performance")    
        print("[FVD    {:.5f}]".format(fvd))
        print("[SSIM   {:.5f}]".format(ssim))
        print("[LPIPS  {:.5f}]".format(lpips))
        print("[PSNR   {:.5f}]".format(psnr))
        
        for i in range(NUM_AUTOREG):
            videos1 = origin_vids[:, cf+pf*i:cf+pf*(i+1)]
            videos2 = result_vids[:, cf+pf*i:cf+pf*(i+1)]
            
            local_ssim = calculate_ssim1(videos1, videos2)[0]
            local_psnr = calculate_psnr1(videos1, videos2)[0]
            local_lpips = calculate_lpips1(videos1, videos2, torch.device("cuda"))[0]
            print(f"{cf+pf*i} ~ {cf+pf*(i+1)-1}th frame prediction performance")
            print("[SSIM   {:.5f}]".format(local_ssim))
            print("[LPIPS  {:.5f}]".format(local_lpips))
            print("[PSNR   {:.5f}]".format(local_psnr))

        return {
            'global_step': global_step,
            'metrics/fvd': fvd,
            'metrics/ssim': ssim,
            'metrics/psnr': psnr,
            'metrics/lpips': lpips,
        }

    
if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args.config)
    
    print(cfg)
    main(cfg, args.valid)
