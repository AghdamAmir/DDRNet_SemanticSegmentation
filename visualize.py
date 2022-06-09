from models.ddrnet import getDDRNet
from utils.config import get_cfg
import torch
import argparse
from torch.nn import Sigmoid
import numpy as np
from matplotlib import pyplot as plt
from utils.DRIVE_utils import ImageFolder



def visualize(config, weight_path, mode='test'):

    model = getDDRNet(cfg=config)
    model.load_state_dict(torch.load(weight_path, map_location=config.TEST.DEVICE))
    model = model.cuda() if config.TEST.DEVICE=='cuda' else model
    model = model.eval()

    dataset = ImageFolder(cfg=config, mode=mode)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.TEST.NUM_WORKERS)
    
    images_fig = plt.figure(figsize= (3, 2))

    for i, (img, gt, original_img) in enumerate(test_loader):
        img = img.cuda() if config.TEST.DEVICE=='cuda' else img
        with torch.no_grad():
            pred = model(img)
        pred = Sigmoid()(pred[0]).squeeze(0).cpu().numpy()
            
        mask = gt[0].squeeze(0).cpu().numpy()
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        ax1 = images_fig.add_axes([1, i, 1, 1])
        ax2 = images_fig.add_axes([2, i, 1, 1])
        ax3 = images_fig.add_axes([3, i, 1, 1])
        ax1.imshow(original_img[0].numpy()[:, :, ::-1])
        ax2.imshow(pred, cmap='gray')
        ax3.imshow(mask, cmap='gray')
    plt.show()
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing segmentation network')
    parser.add_argument('--cfg',
                        help='experiment config file address',
                        default="configs/ddrnet_DRIVE.yaml",
                        type=str)
    parser.add_argument('--weight',
                        help='path to the trained weights',
                        default="weights/best_loss.pth",
                        type=str)
    parser.add_argument('--mode',
                        help='visualization output mode',
                        default="test",
                        type=str)                    
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    visualize(config=cfg, weight_path=args.weight, mode=args.mode)