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
    input = '''[EPOCH]: 1 | [TRAIN LOSS]: 3.9829 | [TRAIN ACC]: 0.2702 | [VAL LOSS]: 4.6819 | [VAL ACC]: 0.1343
[EPOCH]: 2 | [TRAIN LOSS]: 2.9989 | [TRAIN ACC]: 0.3863 | [VAL LOSS]: 4.7301 | [VAL ACC]: 0.1364
[EPOCH]: 3 | [TRAIN LOSS]: 2.6952 | [TRAIN ACC]: 0.4228 | [VAL LOSS]: 4.7886 | [VAL ACC]: 0.1383
[EPOCH]: 4 | [TRAIN LOSS]: 2.5221 | [TRAIN ACC]: 0.4435 | [VAL LOSS]: 4.8651 | [VAL ACC]: 0.1344
[EPOCH]: 5 | [TRAIN LOSS]: 2.4034 | [TRAIN ACC]: 0.4572 | [VAL LOSS]: 4.9551 | [VAL ACC]: 0.1337
[EPOCH]: 6 | [TRAIN LOSS]: 2.3152 | [TRAIN ACC]: 0.4677 | [VAL LOSS]: 5.0161 | [VAL ACC]: 0.1335
[EPOCH]: 7 | [TRAIN LOSS]: 2.2459 | [TRAIN ACC]: 0.4767 | [VAL LOSS]: 5.0783 | [VAL ACC]: 0.1308
[EPOCH]: 8 | [TRAIN LOSS]: 2.1898 | [TRAIN ACC]: 0.4841 | [VAL LOSS]: 5.1319 | [VAL ACC]: 0.1314
[EPOCH]: 9 | [TRAIN LOSS]: 2.1436 | [TRAIN ACC]: 0.4905 | [VAL LOSS]: 5.1709 | [VAL ACC]: 0.1312
[EPOCH]: 10 | [TRAIN LOSS]: 2.1045 | [TRAIN ACC]: 0.4958 | [VAL LOSS]: 5.2264 | [VAL ACC]: 0.1320
[EPOCH]: 11 | [TRAIN LOSS]: 2.0709 | [TRAIN ACC]: 0.5002 | [VAL LOSS]: 5.2663 | [VAL ACC]: 0.1304
[EPOCH]: 12 | [TRAIN LOSS]: 2.0414 | [TRAIN ACC]: 0.5043 | [VAL LOSS]: 5.3043 | [VAL ACC]: 0.1263
[EPOCH]: 13 | [TRAIN LOSS]: 2.0158 | [TRAIN ACC]: 0.5079 | [VAL LOSS]: 5.3301 | [VAL ACC]: 0.1276
[EPOCH]: 14 | [TRAIN LOSS]: 1.9932 | [TRAIN ACC]: 0.5109 | [VAL LOSS]: 5.3481 | [VAL ACC]: 0.1283
[EPOCH]: 15 | [TRAIN LOSS]: 1.9724 | [TRAIN ACC]: 0.5142 | [VAL LOSS]: 5.3878 | [VAL ACC]: 0.1282
[EPOCH]: 16 | [TRAIN LOSS]: 1.9549 | [TRAIN ACC]: 0.5168 | [VAL LOSS]: 5.3886 | [VAL ACC]: 0.1289
[EPOCH]: 17 | [TRAIN LOSS]: 1.9375 | [TRAIN ACC]: 0.5193 | [VAL LOSS]: 5.4184 | [VAL ACC]: 0.1286
[EPOCH]: 18 | [TRAIN LOSS]: 1.9219 | [TRAIN ACC]: 0.5214 | [VAL LOSS]: 5.4806 | [VAL ACC]: 0.1263
[EPOCH]: 19 | [TRAIN LOSS]: 1.9080 | [TRAIN ACC]: 0.5233 | [VAL LOSS]: 5.4674 | [VAL ACC]: 0.1291
[EPOCH]: 20 | [TRAIN LOSS]: 1.8950 | [TRAIN ACC]: 0.5253 | [VAL LOSS]: 5.4911 | [VAL ACC]: 0.1282
[EPOCH]: 21 | [TRAIN LOSS]: 1.8828 | [TRAIN ACC]: 0.5270 | [VAL LOSS]: 5.5550 | [VAL ACC]: 0.1268
[EPOCH]: 22 | [TRAIN LOSS]: 1.8716 | [TRAIN ACC]: 0.5286 | [VAL LOSS]: 5.5759 | [VAL ACC]: 0.1263
[EPOCH]: 23 | [TRAIN LOSS]: 1.8619 | [TRAIN ACC]: 0.5299 | [VAL LOSS]: 5.6122 | [VAL ACC]: 0.1273
[EPOCH]: 24 | [TRAIN LOSS]: 1.8517 | [TRAIN ACC]: 0.5317 | [VAL LOSS]: 5.6158 | [VAL ACC]: 0.1282
[EPOCH]: 25 | [TRAIN LOSS]: 1.8425 | [TRAIN ACC]: 0.5331 | [VAL LOSS]: 5.6417 | [VAL ACC]: 0.1274
[EPOCH]: 26 | [TRAIN LOSS]: 1.8342 | [TRAIN ACC]: 0.5343 | [VAL LOSS]: 5.6690 | [VAL ACC]: 0.1273
[EPOCH]: 27 | [TRAIN LOSS]: 1.8260 | [TRAIN ACC]: 0.5357 | [VAL LOSS]: 5.6901 | [VAL ACC]: 0.1257
[EPOCH]: 28 | [TRAIN LOSS]: 1.8185 | [TRAIN ACC]: 0.5369 | [VAL LOSS]: 5.7006 | [VAL ACC]: 0.1253
[EPOCH]: 29 | [TRAIN LOSS]: 1.8113 | [TRAIN ACC]: 0.5381 | [VAL LOSS]: 5.7298 | [VAL ACC]: 0.1258
[EPOCH]: 30 | [TRAIN LOSS]: 1.8051 | [TRAIN ACC]: 0.5391 | [VAL LOSS]: 5.7388 | [VAL ACC]: 0.1249
[EPOCH]: 31 | [TRAIN LOSS]: 1.7988 | [TRAIN ACC]: 0.5401 | [VAL LOSS]: 5.7535 | [VAL ACC]: 0.1248
[EPOCH]: 32 | [TRAIN LOSS]: 1.7923 | [TRAIN ACC]: 0.5413 | [VAL LOSS]: 5.7550 | [VAL ACC]: 0.1271
[EPOCH]: 33 | [TRAIN LOSS]: 1.7861 | [TRAIN ACC]: 0.5424 | [VAL LOSS]: 5.7718 | [VAL ACC]: 0.1269
[EPOCH]: 34 | [TRAIN LOSS]: 1.7805 | [TRAIN ACC]: 0.5435 | [VAL LOSS]: 5.8381 | [VAL ACC]: 0.1256
[EPOCH]: 35 | [TRAIN LOSS]: 1.7752 | [TRAIN ACC]: 0.5443 | [VAL LOSS]: 5.8357 | [VAL ACC]: 0.1251
[EPOCH]: 36 | [TRAIN LOSS]: 1.7696 | [TRAIN ACC]: 0.5452 | [VAL LOSS]: 5.8772 | [VAL ACC]: 0.1274
[EPOCH]: 37 | [TRAIN LOSS]: 1.7647 | [TRAIN ACC]: 0.5459 | [VAL LOSS]: 5.8749 | [VAL ACC]: 0.1252
[EPOCH]: 38 | [TRAIN LOSS]: 1.7599 | [TRAIN ACC]: 0.5469 | [VAL LOSS]: 5.8851 | [VAL ACC]: 0.1268
[EPOCH]: 39 | [TRAIN LOSS]: 1.7553 | [TRAIN ACC]: 0.5475 | [VAL LOSS]: 5.8998 | [VAL ACC]: 0.1267
[EPOCH]: 40 | [TRAIN LOSS]: 1.7510 | [TRAIN ACC]: 0.5481 | [VAL LOSS]: 5.9319 | [VAL ACC]: 0.1260
'''
    train_states = re.findall(
        r'\[EPOCH\]: \d+ \| \[TRAIN LOSS\]: ([0-9.]+) \| \[TRAIN ACC\]: ([0-9.]+) \| \[VAL LOSS\]: ([0-9.]+) \| \[VAL ACC\]: ([0-9.]+)',
        input)
    plot_performance(*[[float(train_state[i]) for train_state in train_states] for i in range(4)])
