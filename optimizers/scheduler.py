from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_scheduler(cfg, optimizer):
    return ReduceLROnPlateau(optimizer=optimizer, patience=cfg.TRAIN.LR_DECAY_PATIENCE, factor=cfg.TRAIN.LR_DECAY_FACTOR,
                            verbose=True)