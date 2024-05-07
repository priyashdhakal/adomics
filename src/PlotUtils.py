import arguments as args 
import os
import logging 
logging.getLogger().setLevel(logging.INFO)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plots_dir = args.PlotUtilArguments.plot_dir
if not os.path.exists(plots_dir):
    logging.info(f"Creating directory {plots_dir}")
    os.makedirs(plots_dir)


def plot_history(history):
    """
    Save two different plots, 1 for loss and another for accuracy from history object 
    returned by tensorflow model.fit
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plots_dir, "accuracy.png"))
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plots_dir, "loss.png"))
    plt.clf()
    logging.info(f"Plots saved to {plots_dir}")
    return True

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    with open(os.path.join(plots_dir, "confusion_matrix.txt"), "w") as f:
        f.write(str(cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.clf()
    logging.info(f"Confusion Matrix saved to {plots_dir}")
    return True