import torch
import os.path as osp


    
def load_checkpoint_with_accelerator(config, accelerator, lr_meter, losses):
    print("Loading checkpoint from", config.checkpoint.resume)
    if osp.exists(config.checkpoint.resume):
        accelerator.load_state(config.checkpoint.resume)
    if osp.exists(config.checkpoint.resume + '.pt'):
        state_dict = torch.load(config.checkpoint.resume + '.pt')
        global_step = state_dict['global_step']
        save_epoch = state_dict['epoch']
        lr_meter.load(state_dict['lr_meter'])
        losses.load(state_dict['losses'])
        print(f"Loaded successfully '{config.checkpoint.resume}' (epoch {state_dict['epoch']}) (global step {state_dict['global_step']})")
    else:
        global_step = 0
        save_epoch = 0
        lr_meter.reset()
        losses.reset()
        print(f'Failed to load checkpoint from {config.checkpoint.resume}')    
    return global_step, save_epoch, lr_meter, losses


def load_ema_checkpoint(config, ema_model):
    print("Loading EMA checkpoint from", config.checkpoint.resume)
    if osp.exists(config.checkpoint.resume):
        checkpoint = torch.load(osp.join(config.checkpoint.resume, 'ema_model.pt'), map_location='cpu')
        if isinstance(checkpoint, dict):
            ema_model.load_state_dict(checkpoint)
        else:
            ema_model = checkpoint
        print(f"Loaded EMA model successfully from '{osp.join(config.checkpoint.resume, 'ema_model.pt')}'")
    else:
        print(f'Failed to load EMA checkpoint from {config.checkpoint.resume}')
    return ema_model

def save_checkpoint_with_accelerator(config, accelerator, global_step, epoch, lr_meter, losses):
    save_path = osp.join(config.checkpoint.output, f'vdm_steps_{global_step}')
    accelerator.save_state(save_path)
    
    save_path_file = save_path + ".pt"
    accelerator.save({
        'global_step': global_step,
        'epoch': epoch,
        'lr_meter': lr_meter.ckpt(),
        'losses': losses.ckpt(),
    }, save_path_file)
    print(f"Saved checkpoint to '{save_path}' (epoch {epoch}) (global step {global_step})")
    
def load_checkpoint_for_inference(config, accelerator):
    print("Loading checkpoint from", config.checkpoint.resume)
    if osp.exists(config.checkpoint.resume):
        accelerator.load_state(config.checkpoint.resume)
    else:
        ValueError("No checkpoint found.")
        
def save_ema_checkpoint(config, ema_model, global_step, epoch):
    save_path = osp.join(config.checkpoint.output, f'vdm_steps_{global_step}/ema_model.pt')
    torch.save(ema_model.state_dict(), save_path)
    print(f"Saved EMA model to '{save_path}' (epoch {epoch}) (global step {global_step})")
    