import matplotlib.pyplot as plt


def plot_results(train_losses, train_accuracies, val_losses, val_accuracies, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(range(1, epochs + 1), val_losses, label="validation")
    ax1.plot(range(1, epochs+1), train_losses, label="training")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(range(1, epochs + 1), val_accuracies, label="validation")
    ax2.plot(range(1, epochs+1), train_accuracies, label="training")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    plt.show()


