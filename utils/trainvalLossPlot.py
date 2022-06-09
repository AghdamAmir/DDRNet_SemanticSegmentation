import pickle
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Plotting Loss curve')
parser.add_argument('--log',
                        help='experiment config file address',
                        type=str)
args = parser.parse_args()

log = None
with open(args.log, 'rb') as f:
    log = pickle.load(f)

train_loss, val_loss = log

fig = plt.figure(figsize =(10, 6))
ax = fig.add_axes([1, 1, 1, 1])
plt.title(f"Train / Val Loss for {len(train_loss)+1} Epochs")
train_plot, = ax.plot(range(len(train_loss)), train_loss)
val_plot, = ax.plot(range(len(val_loss)), val_loss)
ax.legend([train_plot, val_plot], ['train', 'val'])
plt.ylabel("Loss", fontsize=15, labelpad=8)
plt.xlabel("Epoch", fontsize=15, labelpad=8)
plt.show()