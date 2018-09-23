import matplotlib.pyplot as plt

def plot_tvsv(training_accuracy,validation_accuracy,batch_size,it, ifsave):
    #plot the training and validation accuracy over iteration
    batch=[(i+1)*batch_size for i in range(len(training_accuracy))]
    plt.plot(batch,training_accuracy,'b-',label="training accuracy")
    plt.plot(batch,validation_accuracy,'r-',label="validation accuracy")
    plt.legend("training&validation accuracy plot")
    plt.xlabel("batch number")
    plt.show()
    plt.savefig("accuracy_fig{}.png".format(it))

def plot_accuracy_over_iteration(training_accuracy,validation_accuracy,it, ifsave = False):
    #training_accuracy,validation_accuracy being arrays, it the iteration times
    x_axis = range(1, it+1)
    plt.plot(x_axis, training_accuracy, 'b', label='training accuracy')
    plt.plot(x_axis, validation_accuracy, 'r', label='validation accuracy')
    plt.xlabel('iteration times')
    plt.show()
    plt.savefig("accuracy_over_iteration")

def plot_accuracy_no_other_over_iteration(training_accuracy,validation_accuracy,it, ifsave = False):
    #training_accuracy,validation_accuracy being arrays, it the iteration times
    x_axis = range(1, it+1)
    plt.plot(x_axis, training_accuracy, 'b', label='training accuracy')
    plt.plot(x_axis, validation_accuracy, 'r', label='validation accuracy')
    plt.xlabel('iteration times')
    plt.show()
    plt.savefig("accuracy_no_other_over_iteration")