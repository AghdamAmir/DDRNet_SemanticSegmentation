from torch.optim import Adam

class AdamOptimizer(Adam):
    def __init__(self, params, cfg, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False) -> None:
        super(AdamOptimizer, self).__init__(params, cfg.TRAIN.LEARNING_RATE, betas, eps, weight_decay, amsgrad)


def get_optimizer(model, cfg):
    return AdamOptimizer(params=model.parameters(), cfg=cfg)

