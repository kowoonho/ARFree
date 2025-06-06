import torch.optim as optim


def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    parameters = set_weight_decay(model, {}, {})

    opt_name = config.optimizer.type
    optimizer = None
    if opt_name == 'AdamW':
        optimizer = optim.AdamW(
            parameters,
            **config.optimizer.params
        )
    else:
        raise ValueError(f'Unsupported optimizer: {opt_name}')

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin