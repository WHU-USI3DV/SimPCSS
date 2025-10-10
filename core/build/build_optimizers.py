import torch

OPTIMIZER_SET = ['SGD', 'Adam']
SCHEDULER_SET = ['ExponentialLR', 'CosineAnnealingLR', 'CyclicLR']

def get_optimizer(cfg, *models, lr=None):
    assert cfg.pipeline.optimizer_name in OPTIMIZER_SET, f"{cfg.pipeline.optimizer_name} is not supported"
    model_param_list = []
    for model in models:
        model_param_list.extend(model.parameters())

    if cfg.pipeline.optimizer_name == 'SGD':
        SGD_lr = lr if lr is not None else cfg.pipeline.optimizer_SGD.lr
        optimizer = torch.optim.SGD(
            model_param_list,
            lr = SGD_lr,
            momentum=cfg.pipeline.optimizer_SGD.momentum,
            weight_decay=cfg.pipeline.optimizer_SGD.weight_decay,
            nesterov=True
        )

    elif cfg.pipeline.optimizer_name == 'Adam':
        Adam_lr = lr if lr is not None else cfg.pipeline.optimizer_Adam.lr
        optimizer = torch.optim.Adam(
            model_param_list,
            lr = Adam_lr
        )
    
    return optimizer

def get_scheduler(cfg, optimizer):
    assert cfg.pipeline.scheduler in SCHEDULER_SET, f"{cfg.pipeline.scheduler} is not supported"
    if cfg.pipeline.scheduler == 'ExponentialLR':
        gamma = 0.98
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if cfg.pipeline.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)
    if cfg.pipeline.scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                            base_lr=cfg.pipeline.optimizer_SGD.lr / 10000,
                                                            max_lr=cfg.pipeline.optimizer_SGD.lr,
                                                            step_size_up=5,
                                                            mode="triangular2")
    
    return scheduler