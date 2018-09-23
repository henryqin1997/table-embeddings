import matplotlib.pyplot as plt

def plot_tvsv(training_accuracy,validation_accuracy,batch_size,it):
    #plot the training vs. validation accuracy, to be implemented
    batch=[(i+1)*batch_size for i in range(len(training_accuracy))]
    plt.plot(batch,training_accuracy,'b-',label="training accuracy")
    plt.plot(batch,validation_accuracy,'r-',label="validation accuracy")
    plt.legend("training&validation accuracy plot")
    plt.xlabel("batch number")
    plt.show()
    plt.savefig("accuracy_fig{}.png".format(it))
    return 0