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
    input = '''[EPOCH]: 1 | [TRAIN LOSS]: 4.7213 | [TRAIN ACC]: 0.0000 | [VAL LOSS]: 4.1669 | [VAL ACC]: 0.0000
[EPOCH]: 2 | [TRAIN LOSS]: 4.0376 | [TRAIN ACC]: 0.0210 | [VAL LOSS]: 3.9788 | [VAL ACC]: 0.0030
[EPOCH]: 3 | [TRAIN LOSS]: 3.7949 | [TRAIN ACC]: 0.0461 | [VAL LOSS]: 3.9143 | [VAL ACC]: 0.0013
[EPOCH]: 4 | [TRAIN LOSS]: 3.6280 | [TRAIN ACC]: 0.0680 | [VAL LOSS]: 3.9103 | [VAL ACC]: 0.0019
[EPOCH]: 5 | [TRAIN LOSS]: 3.5017 | [TRAIN ACC]: 0.0782 | [VAL LOSS]: 3.9078 | [VAL ACC]: 0.0020
[EPOCH]: 6 | [TRAIN LOSS]: 3.4014 | [TRAIN ACC]: 0.0861 | [VAL LOSS]: 3.9068 | [VAL ACC]: 0.0018
[EPOCH]: 7 | [TRAIN LOSS]: 3.3194 | [TRAIN ACC]: 0.0971 | [VAL LOSS]: 3.8966 | [VAL ACC]: 0.0025
[EPOCH]: 8 | [TRAIN LOSS]: 3.2484 | [TRAIN ACC]: 0.1055 | [VAL LOSS]: 3.8874 | [VAL ACC]: 0.0024
[EPOCH]: 9 | [TRAIN LOSS]: 3.1869 | [TRAIN ACC]: 0.1125 | [VAL LOSS]: 3.8790 | [VAL ACC]: 0.0026
[EPOCH]: 10 | [TRAIN LOSS]: 3.1323 | [TRAIN ACC]: 0.1189 | [VAL LOSS]: 3.8727 | [VAL ACC]: 0.0027
[EPOCH]: 11 | [TRAIN LOSS]: 3.0830 | [TRAIN ACC]: 0.1250 | [VAL LOSS]: 3.8652 | [VAL ACC]: 0.0029
[EPOCH]: 12 | [TRAIN LOSS]: 3.0385 | [TRAIN ACC]: 0.1309 | [VAL LOSS]: 3.8536 | [VAL ACC]: 0.0029
[EPOCH]: 13 | [TRAIN LOSS]: 2.9972 | [TRAIN ACC]: 0.1348 | [VAL LOSS]: 3.8446 | [VAL ACC]: 0.0038
[EPOCH]: 14 | [TRAIN LOSS]: 2.9600 | [TRAIN ACC]: 0.1393 | [VAL LOSS]: 3.8383 | [VAL ACC]: 0.0047
[EPOCH]: 15 | [TRAIN LOSS]: 2.9245 | [TRAIN ACC]: 0.1439 | [VAL LOSS]: 3.8338 | [VAL ACC]: 0.0059
[EPOCH]: 16 | [TRAIN LOSS]: 2.8924 | [TRAIN ACC]: 0.1475 | [VAL LOSS]: 3.8326 | [VAL ACC]: 0.0084
[EPOCH]: 17 | [TRAIN LOSS]: 2.8620 | [TRAIN ACC]: 0.1499 | [VAL LOSS]: 3.8350 | [VAL ACC]: 0.0093
[EPOCH]: 18 | [TRAIN LOSS]: 2.8329 | [TRAIN ACC]: 0.1523 | [VAL LOSS]: 3.8455 | [VAL ACC]: 0.0101
[EPOCH]: 19 | [TRAIN LOSS]: 2.8063 | [TRAIN ACC]: 0.1545 | [VAL LOSS]: 3.8520 | [VAL ACC]: 0.0108
[EPOCH]: 20 | [TRAIN LOSS]: 2.7800 | [TRAIN ACC]: 0.1571 | [VAL LOSS]: 3.8580 | [VAL ACC]: 0.0115
[EPOCH]: 21 | [TRAIN LOSS]: 2.7558 | [TRAIN ACC]: 0.1595 | [VAL LOSS]: 3.8693 | [VAL ACC]: 0.0119
[EPOCH]: 22 | [TRAIN LOSS]: 2.7329 | [TRAIN ACC]: 0.1616 | [VAL LOSS]: 3.8804 | [VAL ACC]: 0.0126
[EPOCH]: 23 | [TRAIN LOSS]: 2.7109 | [TRAIN ACC]: 0.1640 | [VAL LOSS]: 3.8831 | [VAL ACC]: 0.0129
[EPOCH]: 24 | [TRAIN LOSS]: 2.6894 | [TRAIN ACC]: 0.1662 | [VAL LOSS]: 3.8898 | [VAL ACC]: 0.0132
[EPOCH]: 25 | [TRAIN LOSS]: 2.6689 | [TRAIN ACC]: 0.1694 | [VAL LOSS]: 3.8983 | [VAL ACC]: 0.0146
[EPOCH]: 26 | [TRAIN LOSS]: 2.6492 | [TRAIN ACC]: 0.1723 | [VAL LOSS]: 3.9058 | [VAL ACC]: 0.0152
[EPOCH]: 27 | [TRAIN LOSS]: 2.6306 | [TRAIN ACC]: 0.1759 | [VAL LOSS]: 3.9129 | [VAL ACC]: 0.0160
[EPOCH]: 28 | [TRAIN LOSS]: 2.6130 | [TRAIN ACC]: 0.1789 | [VAL LOSS]: 3.9182 | [VAL ACC]: 0.0153
[EPOCH]: 29 | [TRAIN LOSS]: 2.5949 | [TRAIN ACC]: 0.1813 | [VAL LOSS]: 3.9276 | [VAL ACC]: 0.0162
[EPOCH]: 30 | [TRAIN LOSS]: 2.5777 | [TRAIN ACC]: 0.1843 | [VAL LOSS]: 3.9331 | [VAL ACC]: 0.0155
[EPOCH]: 31 | [TRAIN LOSS]: 2.5613 | [TRAIN ACC]: 0.1871 | [VAL LOSS]: 3.9415 | [VAL ACC]: 0.0153
[EPOCH]: 32 | [TRAIN LOSS]: 2.5456 | [TRAIN ACC]: 0.1892 | [VAL LOSS]: 3.9409 | [VAL ACC]: 0.0135
[EPOCH]: 33 | [TRAIN LOSS]: 2.5300 | [TRAIN ACC]: 0.1914 | [VAL LOSS]: 3.9523 | [VAL ACC]: 0.0145
[EPOCH]: 34 | [TRAIN LOSS]: 2.5154 | [TRAIN ACC]: 0.1936 | [VAL LOSS]: 3.9600 | [VAL ACC]: 0.0141
[EPOCH]: 35 | [TRAIN LOSS]: 2.5008 | [TRAIN ACC]: 0.1961 | [VAL LOSS]: 3.9689 | [VAL ACC]: 0.0141
[EPOCH]: 36 | [TRAIN LOSS]: 2.4867 | [TRAIN ACC]: 0.1977 | [VAL LOSS]: 3.9778 | [VAL ACC]: 0.0142
[EPOCH]: 37 | [TRAIN LOSS]: 2.4730 | [TRAIN ACC]: 0.1998 | [VAL LOSS]: 3.9848 | [VAL ACC]: 0.0142
[EPOCH]: 38 | [TRAIN LOSS]: 2.4603 | [TRAIN ACC]: 0.2021 | [VAL LOSS]: 3.9921 | [VAL ACC]: 0.0136
[EPOCH]: 39 | [TRAIN LOSS]: 2.4470 | [TRAIN ACC]: 0.2041 | [VAL LOSS]: 4.0024 | [VAL ACC]: 0.0134
[EPOCH]: 40 | [TRAIN LOSS]: 2.4350 | [TRAIN ACC]: 0.2054 | [VAL LOSS]: 4.0099 | [VAL ACC]: 0.0141
[EPOCH]: 41 | [TRAIN LOSS]: 2.4223 | [TRAIN ACC]: 0.2069 | [VAL LOSS]: 4.0199 | [VAL ACC]: 0.0134
[EPOCH]: 42 | [TRAIN LOSS]: 2.4111 | [TRAIN ACC]: 0.2084 | [VAL LOSS]: 4.0333 | [VAL ACC]: 0.0137
[EPOCH]: 43 | [TRAIN LOSS]: 2.3991 | [TRAIN ACC]: 0.2099 | [VAL LOSS]: 4.0356 | [VAL ACC]: 0.0140
[EPOCH]: 44 | [TRAIN LOSS]: 2.3874 | [TRAIN ACC]: 0.2119 | [VAL LOSS]: 4.0475 | [VAL ACC]: 0.0143
[EPOCH]: 45 | [TRAIN LOSS]: 2.3774 | [TRAIN ACC]: 0.2132 | [VAL LOSS]: 4.0552 | [VAL ACC]: 0.0132
[EPOCH]: 46 | [TRAIN LOSS]: 2.3688 | [TRAIN ACC]: 0.2144 | [VAL LOSS]: 4.0735 | [VAL ACC]: 0.0146
[EPOCH]: 47 | [TRAIN LOSS]: 2.3571 | [TRAIN ACC]: 0.2159 | [VAL LOSS]: 4.0772 | [VAL ACC]: 0.0142
[EPOCH]: 48 | [TRAIN LOSS]: 2.3459 | [TRAIN ACC]: 0.2182 | [VAL LOSS]: 4.0909 | [VAL ACC]: 0.0117
[EPOCH]: 49 | [TRAIN LOSS]: 2.3354 | [TRAIN ACC]: 0.2195 | [VAL LOSS]: 4.0994 | [VAL ACC]: 0.0125
[EPOCH]: 50 | [TRAIN LOSS]: 2.3256 | [TRAIN ACC]: 0.2210 | [VAL LOSS]: 4.1130 | [VAL ACC]: 0.0122
'''
    train_states = re.findall(
        r'\[EPOCH\]: \d+ \| \[TRAIN LOSS\]: ([0-9.]+) \| \[TRAIN ACC\]: ([0-9.]+) \| \[VAL LOSS\]: ([0-9.]+) \| \[VAL ACC\]: ([0-9.]+)',
        input)
    plot_performance(*[[float(train_state[i]) for train_state in train_states] for i in range(4)])
