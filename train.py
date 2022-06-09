import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from models.ddrnet import getDDRNet
from utils.config import get_cfg
from optimizers.adam import get_optimizer
from optimizers.scheduler import get_scheduler
from utils.DRIVE_utils import ImageFolder


def train(config):
    
    train_loss_list=[]
    val_loss_list = []

    model = getDDRNet(cfg=config)
    
    loss = None
    if cfg.LOSS_FUNCTION.NAME.lower() == 'bceloss':
      from loss.weighted_BCE import get_loss
      loss = get_loss(cfg=config)
    elif cfg.LOSS_FUNCTION.NAME.lower() == 'diceloss':
      from loss.diceLoss import get_loss
      loss = get_loss(cfg=config)
    else:
      raise "Invalid loss Name!"

    optimizer = get_optimizer(model=model, cfg=config)
    
    scheduler = get_scheduler(cfg=config, optimizer=optimizer)

    if config.TRAIN.DEVICE == 'cuda':
      model = model.cuda()


    train_dataset = ImageFolder(cfg=config, mode='train')
    train_data_loader = DataLoader(
            train_dataset,
            batch_size=config.TRAIN.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=config.TRAIN.NUM_WORKERS)
        
    val_dataset = ImageFolder(cfg=config, mode='val')
    val_data_loader = DataLoader(
            val_dataset,
            batch_size=config.VALIDATION.VAL_BATCH_SIZE,
            shuffle=True,
            num_workers=config.VALIDATION.NUM_WORKERS)


    val_epoch_best_loss = config.TRAIN.INITAL_EPOCH_LOSS

    for epoch in range(config.TRAIN.TOTAL_EPOCH):
        train_epoch_loss = 0
        val_epoch_loss = 0

        model = model.train()
        for img, mask, _ in train_data_loader:
            optimizer.zero_grad()
            if config.TRAIN.DEVICE == 'cuda':
              img, mask = img.cuda(), mask.cuda()
            pred = model(img)
            if cfg.LOSS_FUNCTION.NAME.lower() == 'diceloss':
              pred = Sigmoid()(pred)
            train_loss = loss(pred, mask)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss


        model = model.eval()
        for img, mask, _ in val_data_loader:
            with torch.no_grad():
              if config.TRAIN.DEVICE == 'cuda':
                img, mask = img.cuda(), mask.cuda()
              val_pred = model(img)
            if cfg.LOSS_FUNCTION.NAME.lower() == 'diceloss':
              val_pred = Sigmoid()(val_pred)
            val_loss = loss(val_pred, mask)
            val_epoch_loss += val_loss
          
        train_epoch_loss = train_epoch_loss/len(train_data_loader)
        val_epoch_loss = val_epoch_loss/len(val_data_loader)

        train_loss_list.append(train_epoch_loss.item())
        val_loss_list.append(val_epoch_loss.item())

        print('********')
        print('epoch:', epoch)
        print('train_loss:', train_epoch_loss.item())
        print('val_loss:', val_epoch_loss.item())
        print('Learning Rate:', optimizer.param_groups[0]['lr'])
            
        scheduler.step((val_epoch_loss+train_epoch_loss)/2.0)

        if [g for g in optimizer.param_groups][0]['lr'] < 5e-7:
          break   

        if val_epoch_loss < val_epoch_best_loss:
          val_epoch_best_loss = train_epoch_loss
          torch.save(model.state_dict(), config.TRAIN.OUTPUT_WEIGHTS_DIR + "/best_loss.pth")

    print('Finish!')
    return [train_loss_list, val_loss_list]


if __name__ == "__main__":
  import pickle
  from datetime import datetime
  parser = argparse.ArgumentParser(description='Training segmentation network')
  parser.add_argument('--cfg',
                        help='experiment config file address',
                        default="configs/ddrnet_DRIVE.yaml",
                        type=str)
  args = parser.parse_args()
  cfg = get_cfg()
  cfg.merge_from_file(args.cfg)
  cfg.freeze()
  print(cfg)
  train_log = train(config=cfg)
  with open("logs/train_log"+datetime.today().strftime('%Y-%m-%d_%H:%M:%S'), "wb") as tl:  
    pickle.dump(train_log, tl)
