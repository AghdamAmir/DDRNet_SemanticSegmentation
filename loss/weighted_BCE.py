import torch.nn as nn
import torch

class WeightedBCEWithLogits(nn.BCEWithLogitsLoss):
    def __init__(self, cfg) -> None:
        super(WeightedBCEWithLogits, self).__init__()
        self.pos_weight = torch.full(size=cfg.DATASET.IMAGE_SIZE, 
                                            fill_value=cfg.LOSS_FUNCTION.POS_CLASS_WEIGHT, device=cfg.TRAIN.DEVICE)


def get_loss(cfg):
    return WeightedBCEWithLogits(cfg=cfg)