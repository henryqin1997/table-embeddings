import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt
import re


def plot_performance(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend(loc="upper right")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="val")
    plt.legend(loc="lower right")

    # Save figure
    plt.savefig("lstm/performance.png")


if __name__ == '__main__':
    input = '''[EPOCH]: 1 | [TRAIN LOSS]: 3.7990 | [TRAIN ACC]: 0.0533 | [VAL LOSS]: 3.7320 | [VAL ACC]: 0.0076
[EPOCH]: 2 | [TRAIN LOSS]: 3.1847 | [TRAIN ACC]: 0.1186 | [VAL LOSS]: 3.6186 | [VAL ACC]: 0.0094
[EPOCH]: 3 | [TRAIN LOSS]: 2.9402 | [TRAIN ACC]: 0.1429 | [VAL LOSS]: 3.6003 | [VAL ACC]: 0.0143
[EPOCH]: 4 | [TRAIN LOSS]: 2.7785 | [TRAIN ACC]: 0.1615 | [VAL LOSS]: 3.6340 | [VAL ACC]: 0.0234
[EPOCH]: 5 | [TRAIN LOSS]: 2.6607 | [TRAIN ACC]: 0.1752 | [VAL LOSS]: 3.6667 | [VAL ACC]: 0.0246
[EPOCH]: 6 | [TRAIN LOSS]: 2.5643 | [TRAIN ACC]: 0.1865 | [VAL LOSS]: 3.6946 | [VAL ACC]: 0.0274
[EPOCH]: 7 | [TRAIN LOSS]: 2.4827 | [TRAIN ACC]: 0.1963 | [VAL LOSS]: 3.7522 | [VAL ACC]: 0.0279
[EPOCH]: 8 | [TRAIN LOSS]: 2.4142 | [TRAIN ACC]: 0.2066 | [VAL LOSS]: 3.8057 | [VAL ACC]: 0.0332
[EPOCH]: 9 | [TRAIN LOSS]: 2.3533 | [TRAIN ACC]: 0.2143 | [VAL LOSS]: 3.8838 | [VAL ACC]: 0.0337
[EPOCH]: 10 | [TRAIN LOSS]: 2.3023 | [TRAIN ACC]: 0.2212 | [VAL LOSS]: 3.9310 | [VAL ACC]: 0.0349
'''
    train_states = re.findall(
        r'\[EPOCH\]: \d+ \| \[TRAIN LOSS\]: ([0-9.]+) \| \[TRAIN ACC\]: ([0-9.]+) \| \[VAL LOSS\]: ([0-9.]+) \| \[VAL ACC\]: ([0-9.]+)',
        input)
    plot_performance(*[[float(train_state[i]) for train_state in train_states] for i in range(4)])
