from models.ddrnet import getDDRNet
from utils.config import get_cfg
import torch
import argparse
from torch.nn import Sigmoid
import numpy as np
import sklearn.metrics as metrics
from utils.DRIVE_utils import ImageFolder


def test(config, weight_path):
    def accuracy(pred_mask, label):
        '''
        acc=(TP+TN)/(TP+FN+TN+FP)
        '''
        pred_mask = pred_mask.astype(np.uint8)
        TP, FN, TN, FP = [0, 0, 0, 0]
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i][j] == 1:
                    if pred_mask[i][j] == 1:
                        TP += 1
                    elif pred_mask[i][j] == 0:
                        FN += 1
                elif label[i][j] == 0:
                    if pred_mask[i][j] == 1:
                        FP += 1
                    elif pred_mask[i][j] == 0:
                        TN += 1
        acc = (TP + TN) / (TP + FN + TN + FP)
        sen = TP / (TP + FN)
        return acc, sen
    
    model = getDDRNet(cfg=config)
    model.load_state_dict(torch.load(weight_path, map_location=config.TEST.DEVICE))
    model = model.cuda() if config.TEST.DEVICE=='cuda' else model
    model = model.eval()

    for mode in ['test', 'val', 'testval', 'train']:
        dataset = ImageFolder(cfg=config, mode=mode)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.TEST.NUM_WORKERS)
        
        total_acc = []
        total_sen = []
        total_auc = []
        for img, gt, _ in test_loader:
            img = img.cuda() if config.TEST.DEVICE=='cuda' else img
            with torch.no_grad():
                pred = model(img)
            pred = Sigmoid()(pred[0]).squeeze(0).cpu().numpy()
            
            mask = gt[0].squeeze(0).cpu().numpy()
            mask[mask >= 0.5] = 1
            mask[mask <= 0.5] = 0
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
                
            pred = pred.astype(np.uint8)
            mask = mask.astype(np.uint8)
            
            total_auc.append(metrics.roc_auc_score(mask.flatten(), pred.flatten()))
            acc, sen = accuracy(pred, mask)
            total_acc.append(acc)
            total_sen.append(sen)

        print(f"Mean {mode} Accuracy: {np.mean(total_acc)}, Accuracy Std. Deviation: {np.std(total_acc)}")
        print(f"Mean {mode} Sensivity: {np.mean(total_sen)}, Sensivity Std. Deviation: {np.std(total_sen)}")
        print(f"Mean {mode} AUC Score: {np.mean(total_auc)}, AUC Score Std. Deviation: {np.std(total_auc)}")


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
    args = parser.parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    test(config=cfg, weight_path=args.weight)